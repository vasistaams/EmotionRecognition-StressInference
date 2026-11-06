# Audit and open facts

This audit records what is actually implemented in the repository as of the current working tree and cross-checks it against the saved benchmark outputs in `results/`.

## A. Key implementation locations

- Swin backbone: `models/swin_fane.py` defines `SwinFANEEncoder` and builds the Swin backbone via `build_swin_backbone(...)` in `models/swin_fane.py:252` and `models/swin_fane.py:284-314`.
- FANE module: `models/fane_module.py` defines `FANEModule` and `FANEModuleLite` (`models/fane_module.py:37`, `models/fane_module.py:188`).
- ARWM modes (`off`, `fixed`, `adaptive`, `context_aware`): the implementation is centered in `models/swin_fane.py:36-202` (`AdaptiveRegionWeightingModule`) and `models/context_aware_arwm.py:10-115` (`ContextAwareRegionWeightingModule`). The mode is resolved by `utils/configuration.py:27-42` and `utils/configuration.py:44-54`.
- BiLSTM and residual refinement: `models/bilstm_resnet.py:31-79` (`ResidualTemporalBlock`) and `models/bilstm_resnet.py:82-181` (`BiLSTMTemporalModule`). The end-to-end framework is assembled in `models/full_framework.py:27-104`.
- Frame-level classifier head: `models/swin_fane.py:205-250` (`EmotionClassifierHead`).
- Train script: `scripts/train.py:263` (`train(config, args)`) and dataset loading/resume logic around `scripts/train.py:341`.
- Evaluation script: `scripts/evaluate.py:132` (`evaluate_fer2013_official_test`) and the dataset-level evaluation utilities around `scripts/evaluate.py:87`.
- Video inference script: `scripts/infer_video.py:285` (`infer_video`).
- Temporal analysis: `utils/temporal_analysis.py` and the inference pipeline in `scripts/infer_video.py` call the temporal utilities; the saved JSON/CSV artifacts under `results/.../temporal_analysis.json` are the measured outputs.
- Latency benchmark: `scripts/benchmark_latency.py:197` (`run_latency_benchmark`), with the default benchmark paths defined at `scripts/benchmark_latency.py:24-48`.

## B. FER2013 loading and split construction

- FER2013 class order is defined in `data/fer2013_dataset.py:41` as `['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']`.
- `FER2013_FOLDER_TO_LABEL` is defined in `data/fer2013_dataset.py:44-51`, mapping `angry, disgust, fear, happy, sad, surprise, neutral` to indices 0..6.
- The dataset builder is `build_fer2013_dataloaders` in `data/fer2013_dataset.py:222-291`.
- Validation split construction: the code iterates over each class, shuffles the class-specific sample list with `np.random.default_rng(seed)`, takes `val_count = int(round(len(class_samples) * validation_ratio))`, and then builds `train_samples` and `val_samples` from the original training-folder images (`data/fer2013_dataset.py:264-287`).
- The official FER2013 test folder is left untouched and is used as `test_ds` with `split='test'` (`data/fer2013_dataset.py:289-291`).
- The config states `split_ratios: [0.7, 0.15, 0.15]` and `subject_independent: true` in `configs/config_arwm_50.yaml:8-23`.
- Therefore the validation set is created from the training split only; the test set is the official 7,178-image FER2013 folder. This is exactly the repository's intended controlled evaluation setup, and it preserves a train/val/test separation without test-set leakage.
- The repository also explicitly checks that the official test folder is the held-out split and not patched from training images; no duplicate-image logic is used to merge splits.

## C. Training hyperparameters

These values are taken from `configs/config_arwm_50.yaml` (`dataset`, `model`, `training` blocks) and are consistent with the training code in `scripts/train.py`.

- Optimizer: `AdamW` (`configs/config_arwm_50.yaml:102-123`)
- Learning rate: `0.0001` (`configs/config_arwm_50.yaml:104`)
- Scheduler: `StepLR` (`configs/config_arwm_50.yaml:109`), with `step_size: 15` and `gamma: 0.5` (`configs/config_arwm_50.yaml:110-111`)
- Weight decay: `0.0001` (`configs/config_arwm_50.yaml:105`)
- Augmentation: `horizontal_flip`, `random_crop`, `color_jitter`, `rotation_degrees: 15`, `gaussian_blur`, etc., under `augmentation.train` in `configs/config_arwm_50.yaml:25-42`
- Loss: `CrossEntropyLoss` (training code uses `criterion = nn.CrossEntropyLoss` and `criterion(output['logits'].float(), labels)` in `scripts/train.py:94-123`)
- Class weighting: `class_weighted_loss: true` (`configs/config_arwm_50.yaml:122`), with `compute_inverse_frequency_weights(...)` in `scripts/train.py:54-68`
- Label smoothing: `0.1` (`configs/config_arwm_50.yaml:123`)
- Epochs: `num_epochs: 50` (`configs/config_arwm_50.yaml:116-118`)
- Batch size: `32` (`configs/config_arwm_50.yaml:115`)

## D. Exact ARWM implementation details

- The ARWM logic is implemented in `models/swin_fane.py:36-202`.
- Token-grid size: the mask builder explicitly documents a 160x160 input mapped to a 5x5 token grid via a Swin 4x patch and 4 downsampling stages (`models/swin_fane.py:88-97`). This is the intended `5 x 5` token map.
- Mask definitions: the geometry is defined in `_make_region_mask(...)` for `eye` (upper-middle bilateral eyes), `eyebrow` (slightly above the eye line), and `mouth` (lower-center region) (`models/swin_fane.py:98-136`).
- MLP input and hidden sizes: `self.region_mlp = nn.Sequential(nn.Linear(feature_dim * 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))` (`models/swin_fane.py:73-78`). The input is the concatenated region features across the three regions, so it is `3 * feature_dim`, with hidden dim `128` by default.
- Fusion coefficient: the code sets `self.fusion_weight = float(fusion_weight)` and the Swin-FANE fusion is `z = z + self.region_attention.fusion_weight * z_region` (`models/swin_fane.py:68`, `models/swin_fane.py:410-418`). The config value is `fusion_weight: 0.3` (`configs/config_arwm_50.yaml:68`).
- Where fusion happens: after `z_swin + z_fane` in `SwinFANEEncoder.forward`; ARWM acts as an additive branch before the classifier (`models/swin_fane.py:393-422`).
- Parameter count of the ARWM branch: the adaptive branch adds the region MLP, projection, and mask-based pooling logic. The concrete branch parameter count is not hard-coded in the code as a single number, but the runtime benchmark output includes the saved model parameter count (e.g., `model_parameter_count` around 103,031,488 in `results/inference_benchmark/context_aware/repeated_20261105_211334/benchmark_environment.json`). The ARWM-specific hidden structure is defined by `feature_dim * 3 -> hidden_dim -> 3` and is part of that total.

## E. Benchmark checkpoint and why the folder is named `context_aware`

The benchmark at `results/inference_benchmark/context_aware/repeated_20261105_211334/` uses the checkpoint at:

- `checkpoints/arwm_comparison_2026/context_aware/best_model.pth`
- resolved config: `results/arwm_comparison_2026/context_aware/resolved_config.yaml`

Evidence: `results/inference_benchmark/context_aware/repeated_20261105_211334/benchmark_environment.json` records both `"checkpoint": ".../checkpoints/arwm_comparison_2026/context_aware/best_model.pth"` and `"configuration": ".../results/arwm_comparison_2026/context_aware/resolved_config.yaml"`.

The directory name is `context_aware` because the saved configuration and model mode are explicitly `context_aware` in the benchmark metadata and in the config loader (`utils/configuration.py:19-42`, `models/context_aware_arwm.py:10-16`), not because of a generic alias or a different training run.

## F. Batch-1 latency benchmark and timing scope

The saved benchmark summary for the context-aware run records:

- `results/inference_benchmark/context_aware/repeated_20261105_211334/repeated_benchmark_summary.md` states `Batch-size-1 inference latency | 86.1435 | ms` and the JSON summary records `mean_of_repetition_means: 86.14348333231949`.
- The metadata explicitly says: `"timing_scope": "Only model.forward_frame on a preprocessed tensor of shape (1, 3, H, W); model and detector loading, face detection, preprocessing, and device transfer are excluded."` in `benchmark_environment.json`.
- The same file also records `"cuda_synchronized_before_and_after_each_timed_forward": true` and `"preprocessing": { ... "device_input_transfer_ms": ... }`.

This means the `86.1 ms` batch-1 value is a model-only forward latency benchmark, not end-to-end face preprocessing + transfer + inference. The saved metadata does not support the claim that the 11.14/11.72/11.70 ms numbers are from the same measurement definition; they are not the same timing scope and are not from the same benchmark file as the `86.1 ms` batch-1 result. The repository evidence is consistent: the `86.1 ms` figure is the batch-1 model forward time on a preprocessed tensor, whereas the other numbers correspond to different timing contexts or different benchmark runs.

## G. Mean probability volatility of the primary track

The saved per-frame probability trace for the recorded video shows:

- `results/testvideo1/results.json` contains `"mean_probability_volatility_half_l1": 0.16171352800538208`.
- `results/testvideo1/diagnostic_report.json` also contains the same value: `"mean_probability_volatility_half_l1": 0.16171352800538208`.

The correct value is therefore `0.16171352800538208`, which rounds to `0.1617`.

## H. BiLSTM / residual-learning existence and checkpoint state

- A BiLSTM implementation exists in code: `models/bilstm_resnet.py:82-181` (`BiLSTMTemporalModule`). The residual smoothing block is `ResidualTemporalBlock` in `models/bilstm_resnet.py:31-79`, and the framework wires it in when `temporal_model == 'bilstm'` in `models/full_framework.py:52-93`.
- The repository includes both `bilstm` and `bilstm_no_residual` temporal-model options in `models/full_framework.py:71-89`.
- The checkpoint evaluation code uses the saved model state dictionary but does not inspect an explicit temporal summary in the benchmark metadata. In the saved context-aware benchmark, the file `benchmark_environment.json` shows the model is explicitly a context-aware ARWM model, but the benchmark data does not claim a temporal-training checkpoint was selected for the spatial benchmark; the repository does contain BiLSTM/Residual code and a ready-made path to use it in the full framework.
- So: the BiLSTM and residual-learning components exist in code and in the full framework path; they are not absent. The saved benchmark primarily examines the frame-level context-aware ARWM model under the `forward_frame` path and does not change that conclusion.

## I. Class index order and negative set

- Class order is defined in `data/fer2013_dataset.py:41` as: `{0: anger, 1: disgust, 2: fear, 3: happiness, 4: sadness, 5: surprise, 6: neutral}`.
- This matches the paper/label convention: `{'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral'}` and is consistent with `configs/config_arwm_50.yaml:1-20`.
- The config also defines `negative_emotion_indices: [0, 1, 2, 4]` in `configs/config_arwm_50.yaml:14-17`, which matches the NADI negative set semantics.

## Summary

The repository implements and stores the relevant evidence: the Swin-FANE + ARWM spatial encoder, a BiLSTM temporal module plus residual refinement, a FER2013 data loader with the intended train/validation split, and a context-aware benchmark that records the exact checkpoint and timing scope. The audited numbers that matter are consistent with the saved JSON/CSV artifacts.
