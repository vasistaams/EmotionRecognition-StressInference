# Latency reconciliation report

- Checkpoint: `C:\Users\vedag\OneDrive\Desktop\EmotionRecognition-StressInference\checkpoints\arwm_comparison_2026\context_aware\best_model.pth`
- Config: `C:\Users\vedag\OneDrive\Desktop\EmotionRecognition-StressInference\results\arwm_comparison_2026\context_aware\resolved_config.yaml`
- Device: `cuda`
- Warm-up iterations: 50
- Timed iterations: 500
- Synchronization: `torch.cuda.synchronize()` before and after each timed forward call

| method | batch size | mean ms | median ms | p95 ms | min ms | max ms | std ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| forward | 1 | 70.432706 | 68.019900 | 92.833525 | 26.339000 | 120.958400 | 13.092125 |
| forward | 8 | 94.703895 | 91.417550 | 111.541775 | 84.355800 | 255.784700 | 14.198286 |
| forward_frame | 1 | 69.555272 | 65.637400 | 112.964680 | 26.700500 | 153.484500 | 17.408748 |
| forward_frame | 8 | 97.765434 | 91.491650 | 117.738635 | 85.648300 | 339.196100 | 18.880002 |
