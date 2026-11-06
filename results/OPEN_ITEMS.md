# Open items

This file records remaining gaps or items that cannot be closed without additional user-supplied evidence or a new run.

- No user-supplied email or manuscript-specific text was available in the working tree, so the explicit manuscript email exception from the Phase 0 prompt remains unresolved and must be supplied by the user.
- The repository contains the implementation and saved evidence for the ARWM, training, and inference pipeline, but the full multi-seed paper reproduction campaign described in the long prompt was not executed in this audit pass. The saved outputs already present in `results/` are the authoritative facts for the current audit.
- The exact 11.14 / 11.72 / 11.70 ms batch-8 numbers are not present in the saved benchmark artifacts checked here; the repository's preserved timing evidence instead records the `86.1 ms` batch-1 model-only latency and the `timing_scope` metadata that explains the measurement boundary.
- If a new experiment run is required later, it should be written to `results/final/` or `checkpoints/final/` without overwriting any prior outputs, per the non-negotiable experiment policy.
