import unittest
from unittest.mock import patch

import torch
from torch import nn

from models.context_aware_arwm import ContextAwareRegionWeightingModule
from models.full_framework import build_framework
from utils.checkpointing import infer_checkpoint_region_mode, validate_checkpoint_region_mode
from utils.configuration import get_region_attention_mode, load_config


class TinyBackbone(nn.Module):
    native_dim = 8

    def forward(self, images):
        pooled = images.mean(dim=(2, 3))
        features = pooled.unsqueeze(1).expand(-1, 25, -1).repeat(1, 1, 3)[:, :, :8]
        return features.mean(dim=1), features


class ContextAwareArwmTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)

    def make_module(self):
        return ContextAwareRegionWeightingModule(
            feature_dim=8, hidden_dim=5, fusion_weight=0.3
        )

    def test_weights_shape_normalization_and_finiteness(self):
        module = self.make_module()
        features = torch.randn(4, 25, 8)
        _, weights, metadata = module(features)
        self.assertEqual(tuple(weights.shape), (4, 3))
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue((weights >= 0).all())
        self.assertTrue(torch.allclose(weights.sum(1), torch.ones(4), atol=1e-6))
        self.assertEqual(metadata["mode"], "context_aware")

    def test_gradients_reach_context_attention_parameters(self):
        module = self.make_module()
        features = torch.randn(3, 25, 8, requires_grad=True)
        context = torch.randn(3, 8, requires_grad=True)
        representation, weights, _ = module(features, context)
        (representation.square().mean() + weights[:, 0].mean()).backward()
        self.assertIsNotNone(module.local_weight.grad)
        self.assertIsNotNone(module.context_weight.grad)
        self.assertGreater(module.context_weight.grad.abs().sum().item(), 0)
        self.assertIsNotNone(features.grad)
        self.assertIsNotNone(context.grad)

    def test_global_context_changes_attention_for_nondegenerate_parameters(self):
        module = self.make_module().eval()
        with torch.no_grad():
            module.local_weight.zero_()
            module.context_weight.fill_(0.1)
            module.context_weight[1].mul_(2.0)
            module.context_weight[2].mul_(3.0)
            module.score_vector.fill_(1.0)
            module.attention_bias.zero_()
        regions = torch.ones(2, 3, 8)
        contexts = torch.stack((torch.zeros(8), torch.full((8,), 0.1)))
        scores, weights = module.score_regions(regions, contexts)
        self.assertFalse(torch.allclose(scores[0], scores[1]))
        self.assertFalse(torch.allclose(weights[0], weights[1]))

    def test_checkpoint_mode_signature_identifies_context_aware_variant(self):
        checkpoint = {
            "model": {
                "spatial_encoder.region_attention.local_weight": torch.zeros(3, 5, 8),
            }
        }
        self.assertEqual(infer_checkpoint_region_mode(checkpoint), "context_aware")
        self.assertEqual(
            validate_checkpoint_region_mode(
                checkpoint, "context_aware", "context_aware.pth"
            ),
            "context_aware",
        )

    def test_full_encoder_builds_all_configured_modes_and_seven_logits(self):
        for mode in ("off", "fixed", "adaptive", "context_aware"):
            config_path = (
                "configs/arwm/context_aware.yaml"
                if mode == "context_aware"
                else f"configs/arwm/{mode}.yaml"
            )
            config = load_config(config_path)
            config["model"]["swin"]["pretrained"] = False
            config["model"]["classifier"]["embedding_dim"] = 8
            config["model"]["bilstm"]["input_dim"] = 8
            self.assertEqual(get_region_attention_mode(config), mode)
            with patch("models.swin_fane.build_swin_backbone", return_value=TinyBackbone()):
                model = build_framework(config).eval()
                with torch.inference_mode():
                    output = model.forward_frame(torch.randn(2, 3, 16, 16))
            self.assertEqual(tuple(output["logits"].shape), (2, 7))
            self.assertEqual(model.spatial_encoder.region_attention_mode, mode)
            if mode == "context_aware":
                self.assertEqual(tuple(output["region_weights"].shape), (2, 3))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_context_module_cuda_forward(self):
        module = self.make_module().cuda()
        features = torch.randn(2, 25, 8, device="cuda")
        with torch.inference_mode():
            _, weights, _ = module(features)
        self.assertEqual(tuple(weights.shape), (2, 3))
        self.assertTrue(torch.isfinite(weights).all())


if __name__ == "__main__":
    unittest.main()
