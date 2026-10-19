import gc
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from scripts.evaluate_arwm_variants import comparable_training_conditions
from models.swin_fane import AdaptiveRegionWeightingModule, SwinFANEEncoder
from utils.checkpointing import (
    infer_checkpoint_region_mode,
    validate_checkpoint_region_mode,
)
from utils.configuration import get_region_attention_mode, load_config
from utils.face_inference import region_weight_log_record
from utils.logger import TrainingLogger


class TinyBackbone(nn.Module):
    native_dim = 8

    def forward(self, images):
        batch_size = images.shape[0]
        features = images.mean(dim=(2, 3)).unsqueeze(1).expand(-1, 25, -1)
        features = features.repeat(1, 1, 3)[:, :, :8]
        embedding = features.mean(dim=1)
        return embedding, features


class ArwmModeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)

    def make_encoder(self, mode):
        with patch("models.swin_fane.build_swin_backbone", return_value=TinyBackbone()):
            return SwinFANEEncoder(
                pretrained=False,
                embedding_dim=8,
                num_classes=7,
                fane_hidden_dim=4,
                classifier_dropout=0.0,
                region_attention_hidden_dim=4,
                region_attention_mode=mode,
            )

    def test_fixed_weights_and_forward_shapes(self):
        encoder = self.make_encoder("fixed").eval()
        self.assertFalse(hasattr(encoder.region_attention, "region_mlp"))
        with torch.inference_mode():
            output = encoder(torch.randn(3, 3, 16, 16))
        weights = output["region_weights"]
        self.assertEqual(tuple(output["probs"].shape), (3, 7))
        self.assertEqual(tuple(output["embedding"].shape), (3, 8))
        self.assertEqual(tuple(weights.shape), (3, 3))
        self.assertTrue(torch.allclose(weights, torch.full_like(weights, 1 / 3)))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(3)))

    def test_adaptive_weights_normalize_and_depend_on_input(self):
        module = AdaptiveRegionWeightingModule(
            feature_dim=8, hidden_dim=4, mode="adaptive"
        ).eval()
        features = torch.randn(4, 25, 8)
        with torch.inference_mode():
            _, weights, metadata = module(features)
        self.assertEqual(tuple(weights.shape), (4, 3))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(4), atol=1e-6))
        self.assertEqual(metadata["mode"], "adaptive")
        self.assertFalse(torch.allclose(weights[0], weights[1]))

    def test_off_matches_legacy_equation_and_has_no_region_parameters(self):
        encoder = self.make_encoder("off").eval()
        self.assertIsNone(encoder.region_attention)
        self.assertFalse(any("region_attention" in key for key in encoder.state_dict()))
        images = torch.randn(2, 3, 16, 16)
        with torch.inference_mode():
            swin_embedding, features = encoder.backbone(images)
            fane_embedding, _ = encoder.fane(features)
            expected_logits, expected_probs = encoder.classifier(
                swin_embedding + encoder.fane_proj(fane_embedding)
            )
            output = encoder(images)
        self.assertTrue(torch.equal(output["logits"], expected_logits))
        self.assertTrue(torch.equal(output["probs"], expected_probs))
        self.assertIsNone(output["region_weights"])

    def test_legacy_and_new_configuration_resolution(self):
        self.assertEqual(
            get_region_attention_mode({"model": {"region_attention": {"enabled": False}}}),
            "off",
        )
        self.assertEqual(
            get_region_attention_mode({"model": {"region_attention": {"enabled": True}}}),
            "adaptive",
        )
        for mode in ("off", "fixed", "adaptive"):
            config = load_config(f"configs/arwm/{mode}.yaml")
            self.assertEqual(get_region_attention_mode(config), mode)
            self.assertEqual(
                config["paths"]["checkpoint_dir"],
                f"checkpoints/arwm_variants/{mode}",
            )

    def test_checkpoint_mode_detection_and_guard(self):
        self.assertEqual(infer_checkpoint_region_mode({"model": {}}), "off")
        adaptive_state = {
            "model": {
                "spatial_encoder.region_attention.region_proj.0.weight": torch.zeros(1),
                "spatial_encoder.region_attention.region_mlp.0.weight": torch.zeros(1),
            }
        }
        self.assertEqual(infer_checkpoint_region_mode(adaptive_state), "adaptive")
        with self.assertRaisesRegex(ValueError, "mode mismatch"):
            validate_checkpoint_region_mode(adaptive_state, "fixed", "adaptive.pth")
        self.assertEqual(
            validate_checkpoint_region_mode(
                {"experiment": {"region_attention_mode": "fixed"}, "model": {}},
                "fixed",
                "fixed.pth",
            ),
            "fixed",
        )

    def test_training_condition_comparison_ignores_only_mode_and_output_paths(self):
        off = {
            "seed": 42,
            "training_config": {
                "model": {"region_attention": {"mode": "off", "enabled": False}},
                "paths": {"checkpoint_dir": "off", "log_dir": "off"},
                "training": {"epochs": 50},
            },
        }
        fixed = {
            "seed": 42,
            "training_config": {
                "model": {"region_attention": {"mode": "fixed", "enabled": True}},
                "paths": {"checkpoint_dir": "fixed", "log_dir": "fixed"},
                "training": {"epochs": 50},
            },
        }
        different = {
            "seed": 43,
            "training_config": {
                "model": {"region_attention": {"mode": "adaptive", "enabled": True}},
                "paths": {"checkpoint_dir": "adaptive"},
                "training": {"epochs": 50},
            },
        }
        self.assertEqual(
            comparable_training_conditions(off),
            comparable_training_conditions(fixed),
        )
        self.assertNotEqual(
            comparable_training_conditions(off),
            comparable_training_conditions(different),
        )

    def test_region_weight_logging_schema_for_each_mode(self):
        off_record = region_weight_log_record("off")
        self.assertEqual(off_record["region_attention_status"], "disabled")
        self.assertIsNone(off_record["arwm_region_weights"])
        self.assertIsNone(off_record["eye_weight"])

        fixed_record = region_weight_log_record("fixed", [1 / 3, 1 / 3, 1 / 3])
        self.assertEqual(fixed_record["region_attention_status"], "active")
        self.assertEqual(
            fixed_record["arwm_region_weights"],
            {"eye": 1 / 3, "eyebrow": 1 / 3, "mouth": 1 / 3},
        )

        adaptive_record = region_weight_log_record("adaptive", [0.2, 0.3, 0.5])
        self.assertEqual(
            adaptive_record["arwm_region_weights"],
            {"eye": 0.2, "eyebrow": 0.3, "mouth": 0.5},
        )
        with self.assertRaisesRegex(ValueError, "requires actual region weights"):
            region_weight_log_record("adaptive")

    def test_latest_and_best_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            model = nn.Linear(3, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=2, gamma=0.5
            )
            logger = TrainingLogger(
                log_dir=directory,
                experiment_name="checkpoint-test",
                use_tensorboard=False,
                use_wandb=False,
            )
            logger.save_checkpoint(
                model,
                optimizer,
                epoch=1,
                metrics={"accuracy": 0.5},
                checkpoint_dir=directory,
                is_best=True,
                metadata={
                    "region_attention_mode": "fixed",
                    "early_stopping": {
                        "best_value": 0.5,
                        "counter": 0,
                        "should_stop": False,
                    },
                },
                scheduler=scheduler,
                save_epoch_checkpoint=False,
            )
            logger.close()

            latest_path = os.path.join(directory, "latest_model.pth")
            best_path = os.path.join(directory, "best_model.pth")
            self.assertTrue(os.path.isfile(latest_path))
            self.assertTrue(os.path.isfile(best_path))
            self.assertFalse(
                os.path.isfile(os.path.join(directory, "checkpoint_epoch001.pth"))
            )

            checkpoint = torch.load(
                latest_path, map_location="cpu", weights_only=False
            )
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertIn("scheduler", checkpoint)
            self.assertIn("rng_state", checkpoint)
            self.assertEqual(
                validate_checkpoint_region_mode(checkpoint, "fixed", latest_path),
                "fixed",
            )
            restored_model = nn.Linear(3, 2)
            restored_model.load_state_dict(checkpoint["model"], strict=True)
            restored_optimizer = torch.optim.AdamW(
                restored_model.parameters(), lr=0.001
            )
            restored_optimizer.load_state_dict(checkpoint["optimizer"])
            restored_scheduler = torch.optim.lr_scheduler.StepLR(
                restored_optimizer, step_size=2, gamma=0.5
            )
            restored_scheduler.load_state_dict(checkpoint["scheduler"])
            self.assertEqual(
                restored_scheduler.state_dict(), scheduler.state_dict()
            )

    @unittest.skipUnless(
        os.path.exists("checkpoints/fer2013_arwm/best_model.pth"),
        "legacy adaptive checkpoint is unavailable",
    )
    def test_legacy_adaptive_checkpoint_loads_strictly(self):
        from models.full_framework import build_framework

        config = load_config("configs/config_arwm_50.yaml")
        config["model"]["swin"]["pretrained"] = False
        self.assertEqual(get_region_attention_mode(config), "adaptive")
        model = build_framework(config).eval()
        checkpoint = torch.load(
            "checkpoints/fer2013_arwm/best_model.pth",
            map_location="cpu",
            weights_only=False,
        )
        validate_checkpoint_region_mode(
            checkpoint, "adaptive", "checkpoints/fer2013_arwm/best_model.pth"
        )
        incompatible = model.load_state_dict(checkpoint["model"], strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])

    @unittest.skipUnless(
        os.path.exists("results/arwm/pre_change_baseline.json")
        and os.path.exists(
            "checkpoints/fer2013_weighted_freeze3_lr1e4/best_model.pth"
        ),
        "pre-change baseline fixture and baseline checkpoint are unavailable",
    )
    def test_prechange_baseline_outputs_unchanged(self):
        from models.full_framework import build_framework

        with open("results/arwm/pre_change_baseline.json", encoding="utf-8") as file:
            baseline = json.load(file)
        config = load_config("configs/config.yaml")
        config["model"]["swin"]["pretrained"] = False
        model = build_framework(config).eval()
        checkpoint = torch.load(
            "checkpoints/fer2013_weighted_freeze3_lr1e4/best_model.pth",
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        images = torch.linspace(
            -1.0, 1.0, steps=3 * 160 * 160, dtype=torch.float32
        ).reshape(1, 3, 160, 160)
        with torch.inference_mode():
            output = model.forward_frame(images)
        for key in ("embedding", "logits", "probs", "pred_label"):
            expected = torch.tensor(baseline["outputs"][key])
            self.assertTrue(torch.allclose(output[key], expected, rtol=1e-6, atol=1e-7))
        del model, checkpoint
        gc.collect()


if __name__ == "__main__":
    unittest.main()
