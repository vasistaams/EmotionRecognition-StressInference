import unittest
from unittest.mock import patch

import torch
from torch import nn

from models.swin_fane import SwinFANEEncoder


class TinyBackbone(nn.Module):
    native_dim = 8

    def forward(self, images):
        batch = images.shape[0]
        features = images.mean(dim=(2, 3)).unsqueeze(1).expand(-1, 25, -1)
        features = features.repeat(1, 1, 3)[:, :, :8]
        return features.mean(dim=1), features


class RegionInitializationRngTests(unittest.TestCase):
    def make_encoder(self, mode):
        with patch(
            "models.swin_fane.build_swin_backbone",
            side_effect=lambda **kwargs: TinyBackbone(),
        ):
            return SwinFANEEncoder(
                pretrained=False,
                embedding_dim=8,
                num_classes=7,
                fane_hidden_dim=4,
                classifier_dropout=0.0,
                region_attention_hidden_dim=4,
                region_attention_mode=mode,
            )

    def test_mode_specific_arwm_does_not_shift_common_initialization(self):
        common_states = {}
        final_rng_states = {}
        for mode in ("off", "fixed", "adaptive", "context_aware"):
            torch.manual_seed(42)
            encoder = self.make_encoder(mode)
            common_states[mode] = {
                "fane": {
                    key: value.detach().clone()
                    for key, value in encoder.fane.state_dict().items()
                },
                "classifier": {
                    key: value.detach().clone()
                    for key, value in encoder.classifier.state_dict().items()
                },
            }
            final_rng_states[mode] = torch.get_rng_state().clone()

        for mode in ("fixed", "adaptive", "context_aware"):
            for component in ("fane", "classifier"):
                for key, expected in common_states["off"][component].items():
                    self.assertTrue(
                        torch.equal(expected, common_states[mode][component][key]),
                        f"{component}.{key} differs in {mode} mode",
                    )
            self.assertTrue(
                torch.equal(final_rng_states["off"], final_rng_states[mode]),
                f"CPU RNG state differs after constructing {mode} mode",
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cpu_model_construction_does_not_change_cuda_rng(self):
        torch.manual_seed(42)
        cuda_state = torch.cuda.get_rng_state().clone()
        self.make_encoder("context_aware")
        self.assertTrue(torch.equal(cuda_state, torch.cuda.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
