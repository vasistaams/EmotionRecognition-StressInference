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

## J. Follow-up

### J.1 Step 1: Requirement matrix for variant evidence

- Wrote the matrix to `results/final/REQUIREMENTS_MATRIX.md`.
- Key evidence included:
  - `results/arwm/off_20/run_summary.json`: `best_checkpoint_validation_metrics.accuracy = 71.22619600557361`, `best_validation_accuracy = 71.22619600557361`.
  - `results/arwm/fixed_20/run_summary.json`: `best_validation_accuracy = 71.66744078030655`.
  - `results/arwm/evaluation_20261023_190929/metrics.json`: carries the controlled comparison summary for off/fixed/adaptive with `trainable_parameters` values `101192640`, `102244288`, and `102638019` respectively.
  - `results/arwm_comparison_2026/context_aware/test_evaluation/evaluation_metrics.json`: `accuracy = 0.7117581499024798`, `balanced_accuracy = 0.7021085857776342`, `macro_f1 = 0.6998816976502785`, `roc_auc_ovr_macro = 0.8701311996537983`.
- Data quality constraint: the legacy `adaptive_*` smoke and legacy evaluation files are not a clean controlled ablation; they are explicitly marked as incomplete/legacy in their own JSON metadata.

### J.2 Step 2: Search for the requested values across results/

- The same value search will be written to `results/final/VALUE_SEARCH_TABLE.md` in the next step.
- The value set requested by the user (`69.98`, `70.87`, `68.80`, `69.38`, `70.99`, `68.51`, `69.50`, `68.46`, `101.19`, `102.24`, `102.64`, `11.14`, `11.72`, `11.70`) is not a single uniform metric stream; it matches the previously saved legacy evaluation outputs and latency-related numbers in different files rather than one single benchmark table.

### J.3 Step 3: Context-aware ARWM exact behavior

- `models/context_aware_arwm.py` defines a global-context-conditioned weighting module; `global_context` defaults to `features.mean(dim=1)` if not supplied.
- The coded equation is:
  - `local = einsum('brc,rhc->brh', region_features, local_weight)`
  - `context = einsum('bc,rhc->brh', global_context, context_weight)`
  - `hidden = tanh(local + context + attention_bias)`
  - `scores = einsum('brh,rh->br', hidden, score_vector)`
  - `weights = softmax(scores, dim=-1)`
- The output has `projected = region_proj(combined)` where `combined = sum(weights * region_features)`, so the projection does exist.
- Parameter counts are mode-specific: legacy adaptive ARWM uses `region_mlp` (`3 * feature_dim -> hidden_dim -> 3`) plus `region_proj`; the context-aware module uses `local_weight`, `context_weight`, `attention_bias`, and `score_vector` with `feature_dim=1024, hidden_dim=128`, plus the same `region_proj` output projection.

### J.4 Step 4: validation_ratio, seed dependence, and MD5 overlap

- The actual validation split ratio implemented in `data/fer2013_dataset.py` is `validation_ratio = 0.15` and the split is done per class with `rng = np.random.default_rng(seed)`.
- The validation subset therefore depends on the seed because it is a seeded, class-wise permutation before slicing the validation count.
- Seeds used in the saved experiments are overwhelmingly `42` (see `configs/config_arwm_50.yaml`, `configs/arwm/comparison_2026_context_aware.yaml`, and the saved run metadata).
- MD5 overlap check on the actual FER2013 files (same logic as the code) gives non-zero hash overlaps between the official test set and train/validation images: `train_overlap = 440`, `validation_overlap = 126`, while the saved path-level inventory reported zero path collisions. This means the risk is bytewise data duplication, not path duplication.

### J.5 Step 5: BiLSTM status and video stress provenance

- `models/bilstm_resnet.py` and `models/full_framework.py` contain a trainable BiLSTM/residual temporal path, but the saved results under `results/testvideo1/temporal_analysis.json` explicitly say `"learned_stress_prediction": false`.
- The `results/testvideo1/temporal_analysis.json` method block shows stress is computed as a proxy from frame-level probabilities: `probability_volatility = 0.5 * L1(P_t - P_(t-1))`, `negative_affect_volatility = abs(r_t - r_(t-1))`, and `stress_proxy = alpha * negative_affect + beta * negative_affect_volatility` with `alpha = 0.7` and `beta = 0.3`.
- Therefore `emotion_transitions`, `volatility`, and `NADI`/stress proxy values in the saved video result are derived from frame-level emotion probabilities, not from a trained BiLSTM output.

### J.6 Step 6: local labeled video emotion/stress datasets

- No local labeled video emotion/stress dataset beyond the FER2013 image dataset and sample demo videos was found in the repository tree.
- The only local video assets under `data/videos/` are `testvideo1.mp4` and `testvideo2.mp4`, and they are not paired with ground-truth emotion/stress labels.

### J.7 Step 7: latency reconciliation

- The synchronized same-checkpoint latency reconciliation was run under `results/final/latency_reconciliation/` for both `forward` and `forward_frame` on batch sizes 1 and 8 with 50 warm-up iterations and 500 timed iterations.
- The saved benchmark summary is in `results/final/latency_reconciliation/latency_reconciliation_summary.md` and the raw JSON/CSV outputs are in the same directory.
- Measured results with CUDA synchronization before and after each timed call (RTX 3050 6GB Laptop GPU):
  - `forward`, batch 1: mean = 70.432706 ms, median = 68.019900 ms, p95 = 92.833525 ms
  - `forward`, batch 8: mean = 94.703895 ms, median = 91.417550 ms, p95 = 111.541775 ms
  - `forward_frame`, batch 1: mean = 69.555272 ms, median = 65.637400 ms, p95 = 112.964680 ms
  - `forward_frame`, batch 8: mean = 97.765434 ms, median = 91.491650 ms, p95 = 117.738635 ms
- These timings are model-only and exclude checkpoint loading, face detection, preprocessing, and transfer; they reflect the same checkpoint (`checkpoints/arwm_comparison_2026/context_aware/best_model.pth`) and resolved config (`results/arwm_comparison_2026/context_aware/resolved_config.yaml`).
