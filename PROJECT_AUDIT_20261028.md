# Repository Audit — 2026-10-28

## Scope and safety

Read-only audit of the existing Swin-FANE / ARWM repository before Context-Aware ARWM training. No training or evaluation was launched. No existing file, directory, checkpoint, log, result, or source file was moved, renamed, deleted, or overwritten. This report is a new file. Findings reflect the repository state observed on 2026-10-28.

## Repository baseline

- Root: `C:\Users\vedag\OneDrive\Desktop\EmotionRecognition-StressInference`
- Branch: `main`; HEAD `63712c7d28c3f6a56dd82ffc871bfb61f97d9596` (`Merge pull request #1 from vasistaams/feature/arwm-ablation`, 2026-10-19).
- Worktree is already substantially dirty: 13 modified tracked files, 64 untracked non-ignored files, and 76,308 ignored files; nothing staged at audit start. Existing user changes and untracked artifacts must be preserved.
- `.gitignore` covers environments, Python caches, data, checkpoints, logs, and results. It does not cover `figures/`, `paper_evidence/`, `research_paper_code/`, or their ZIP archives; broad `git add .` could stage these.
- **Critical source issue:** `README.md` contains unresolved merge-conflict markers (`<<<<<<< HEAD` at line 1, `=======` at line 291, `>>>>>>> origin/main` at line 293). Its two versions are concatenated in one file. Do not choose or discard either side without review. `.gitignore` has no such markers.

## Top-level inventory

Approximate on-disk totals (directory totals are from recursive file inventory; `.git`/`.venv` internals are not enumerated in this report):

| Area | Files | Bytes (approx.) | Role / observation |
|---|---:|---:|---|
| `.git` | 215 | 19,295,653 | Git metadata; keep |
| `.venv` | 39,937 | 5,072,324,180 | Local Python environment; ignored, reproducible but costly to recreate |
| `checkpoints` | 60 | 59,126,575,167 | Training snapshots; preserve all pending explicit selection |
| `configs` | 10 | 13,931 | Base and ARWM experiment YAMLs |
| `data` | 35,897 | 123,193,775 | FER2013 and video inputs; ignored, preserve |
| `figures` | 10 | 3,845,086 | Root-level paper/analysis figures; potentially duplicated in package |
| `logs` | 24 | 585,174 | TensorBoard and run metadata |
| `models` | 12 | 145,277 | Model implementations, including Context-Aware ARWM |
| `paper_evidence` | 166 | 22,013,312 | Evidence snapshot; exact content also in ZIP |
| `research_paper_code` | 16 | 203,497 | Extracted source snapshot; exact content also in ZIP |
| `results` | 221 | 172,637,348 | Evaluation, video inference, experiment and paper-package outputs |
| `scripts` | 20 | 649,564 | Training, evaluation, inference, analysis and packaging scripts |
| `tests` | 6 | 64,179 | ARWM/context/video tests |
| `utils` | 18 | 222,323 | Configuration, checkpoints, inference, metrics and visualization helpers |

Other root files: `.gitignore`, `README.md`, `requirements.txt`, `SWIN_ARWM_Colab_Experiments.ipynb`, `paper_evidence.zip` (12,901,709 bytes), and `research_paper_code.zip` (57,692 bytes). Overall project footprint is about 64.5 GB, dominated by checkpoints and `.venv`.

All visible root entries report Windows `ReparsePoint` attributes in this OneDrive-backed workspace, but PowerShell did not resolve their link targets. Treat link/junction ownership and outside-root targets as **unverified**; do not bulk traverse or move them based on attributes alone. A Python process query showed no `python`/`pythonw` process, but Windows denied the broader process query, so open-handle ownership could not be conclusively checked.

## Code, configuration, and paths

- Context-Aware model implementation is `models/context_aware_arwm.py`; mode/config integration is present in the modified model, train/evaluate/inference and utility files, with `tests/test_context_aware_arwm.py` and `tests/test_arwm_modes.py` untracked. These files are part of the dirty worktree and must be preserved.
- `configs/arwm/context_aware.yaml` inherits `configs/config_arwm_50.yaml`; it sets seed 42, 160-pixel inputs, batch size 8, 20 epochs, mixed precision, and outputs to `checkpoints/context_aware_arwm`, `logs/context_aware_arwm`, and `results/arwm/context_aware`. Those dedicated output directories were absent at audit time, as expected before the first run.
- The `adaptive_controlled_seed42_resume.yaml` config inherits `adaptive_controlled_seed42.yaml`, and redirects checkpoints/logs/results to the resume-specific directories. Its referenced config exists.
- Configuration inheritance is relative to the containing YAML and merged recursively by `utils/configuration.py`.
- `scripts/build_paper_artifacts.py` stages a package and atomically replaces the destination; on exception it recursively removes its staging directory. This script was inspected but not executed. Its cleanup behavior warrants a path-confinement review before running it with a custom `--output`.
- Legacy README commands and paths need review alongside the merge conflict. Prior path inspection found references to `checkpoints/fer2013_arwm/best_model.pth`, which is absent. Resolve docs after selecting the intended README version.

## Experiment artifacts and provenance

- `checkpoints/` contains root baseline snapshots plus ARWM adaptive, controlled/resumed adaptive, fixed-20, and off-20 series. Snapshots include optimizer/scheduler state and are research artifacts; do not delete “older” epochs based only on chronology.
- `logs/` contains corresponding ARWM run logs. No `logs/context_aware_arwm/` existed.
- `results/` contains FER2013 ARWM comparisons/evaluations, checkpoint metric experiments, three real-video result runs (`testvideo1`, `testvideo1_rerun`, `testvideo2`), and `paper_artifacts/` with evaluation, figures, and preexisting figures. Preserve provenance and reports.
- Context-Aware dedicated checkpoint, log, and result outputs were absent, so there is no prior Context-Aware run to archive or clean.
- FER2013 training loader uses a seeded image-level validation split from the train folder and keeps the official test folder held out. There are no subject IDs, so the validation is not subject-independent.
- A prior checkpoint inspection loaded snapshots safely with `weights_only=True` and metadata-only/mmap behavior; epochs and mode/state-key counts aligned with the expected series. No checkpoint content was changed in this audit.

## Duplicate and empty-file findings

Verified exact duplicates:

1. Every entry in `research_paper_code.zip` (16 entries) matches the corresponding extracted file by name and SHA-256.
2. Every entry in `paper_evidence.zip` (166 entries) matches the corresponding extracted file by name and SHA-256.

These pairs may be intentional distribution snapshots. Candidate space recovery is to retain either the ZIP or extracted tree for each pair, only after confirming which delivery form is needed; neither was removed. Root `figures/` appears to overlap with `results/paper_artifacts/figures`; hash-level duplicates were observed, but the complete exact pair list should be generated before any removal. Similarity between `testvideo1` and `testvideo1_rerun` is not sufficient evidence to remove either run.

`paper_evidence/python_packages.txt` is zero bytes. It is also included in the exact ZIP/tree duplicate pair; retain until evidence requirements are checked.

## Proposed cleanup plan (not executed)

### Repair first

1. Resolve the README merge conflict by reviewing the intended documentation and preserving useful content from both sides; then correct stale checkpoint paths and document the Context-Aware mode/output paths.
2. Review `.gitignore` policy for generated root figures and evidence/source bundles. Decide whether these are intended deliverables to commit or should be ignored/relocated; do not blanket-ignore without deciding.
3. Review artifact packaging script path confinement and only then run it, if still needed.

### Optional space recovery after owner approval

- For each ZIP/tree exact duplicate pair, choose the canonical retained form and remove only the redundant copy after confirming recoverability and intended sharing format. Approximate maximum recovered space is bounded by the smaller side of each pair (not the sum of both).
- For root figures versus packaged figures, generate and review a full SHA-256 pair manifest; remove only byte-identical duplicates after confirming that both locations are not separate deliverables.
- Consider `.venv` recreation only as a separate decision: it is about 5.07 GB, ignored, and recreatable from dependencies, but rebuilding may require network access and GPU-compatible wheels. It is not proposed for deletion now.
- Do not delete checkpoints, dataset files, run logs, experiment results, video outputs, evidence, or source snapshots by age or name alone.

## Approval gate

No move, rename, overwrite, or deletion has been performed. Before cleanup, review the README conflict and decide which ZIP/tree copies and figure copies are canonical. Once the exact files and intended retained form are agreed, cleanup can be prepared as a separate, reviewable change. Training remains unstarted.
