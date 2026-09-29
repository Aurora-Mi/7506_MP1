# 224x8 matrix weight decay 0.6 — validation result

This controlled experiment changed only matrix weight decay from 0.5 to 0.6.
Architecture, seed, 24,000 updates, processed training targets, sampling,
learning-rate schedule, MTP, EMA, and validation selection were unchanged.
Development used only the locked train/validation workflow; test was not loaded.

## Results

- Validation-selected checkpoint: step 20,000 EMA.
- Neural/cache BPB before recalibration: `1.463201846898695`.
- Selected calibration: temperature 1.12, cache weight 0.04, theta 20.
- Calibrated BPB: `1.4630333645325382`.
- Selected train-only bigram: alpha 1, weight 0.02.
- Exported CPU FP32 validation BPB: `1.4612034232171305`.
- Checkpoint SHA256: `ceb4beddfc3ce851c5859e57f0b3cf773f541f897cfd3ecdf2b4299e4e467dd4`.
- Inference assets: 36.305 MiB.
- Validation scoring time: 21.101 seconds; recorded baseline 4.921 seconds,
  approximately 4.29x.

The prior WD 0.5 complete candidate scored `1.464390297786845`, so WD 0.6
improves validation by about `0.0031869` BPB without changing inference structure
or asset size. It is the current development candidate. Do not run test until the
method is explicitly frozen and all remaining development has stopped.
