# FER2013 deduplicated split

The CSV manifests list repository-relative image paths, numeric labels, class
names, and SHA-256 hashes of decoded grayscale pixels. The image files are not
copied or modified.

The audit first reproduces the loader's original per-class validation split
(seed 42, rounded 15% holdout) to measure duplicates. For the clean split it
then removes every Training image whose decoded-pixel hash occurs in the
combined official test directory, excludes remaining duplicate-hash groups
with conflicting class labels, retains one deterministic path for each
remaining pixel hash, and creates a new stratified 15% validation holdout
(seed 42). This additionally prevents duplicate content crossing train and
validation. The official test files are unchanged; repeated content and
conflicting labels within test are reported, not removed.

`test_combined.csv` lists all 7,178 test images. The `test_private.csv` and
`test_public.csv` manifests select the 3,589 files in each official split by
their filename prefix.

Regenerate the audit and lists from the local FER2013 folder with:

```powershell
.\.venv\Scripts\python.exe scripts\build_fer2013_deduplicated_split.py
```

Train from these manifests by passing
`--split-manifest reproducibility\fer2013_deduplicated_seed42` to
`scripts\train.py`. Evaluate with the same option and select
`--test-split test_combined` or `--test-split test_private`. Test results must
not be used for training, model selection, or hyperparameter decisions.
