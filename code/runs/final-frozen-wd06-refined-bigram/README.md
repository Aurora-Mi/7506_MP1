# Frozen and tested final predictor — do not modify

Frozen on 2026-09-28 at 12:44:35 UTC+8 after validation-only development.
No further architecture, training, calibration, interpolation, n-gram, or
checkpoint selection changes are permitted after this point.

Validation CPU FP32 BPB: **1.4598314480072423**.
Checkpoint SHA256:
`c8c30df36e1e0b29beb11c36155d3f36020d2f8ad8462ee55d89e25a77db574f`.

Predictor: 224x8, 7-head MHA, MTP training weight 0.2, matrix weight decay 0.6,
step-20,000 EMA weights. Scoring uses temperature 1.11, within-window cache
weight 0.05/theta 12, and train-derived bigram alpha 1/weight 0.02.
Inference assets are 36.305 MiB. Validation scoring took 21.499 seconds versus
the recorded 4.921-second development baseline (about 4.37x).

Required inference modules are `student_bigram.py` and `student.py`. Training,
calibration, bigram search, export, validation result, data hashes, and source
snapshots are retained in this directory as provenance. The original project
data remain outside this snapshot.

The one final CPU FP32 test was completed after the freeze:

- Test BPB: **1.476789395067789**.
- Token perplexity: 21.914321686973548.
- Scored targets: 428,405.
- UTF-8 bytes: 1,292,013.
- Scoring time: 24.190735699958168 seconds.
- Test-result SHA256:
  `801e53edce97f65953e8b7ebbc9fb72ebd170856e3eaf80ed5fc48cfa0476328`.
- Recorded baseline time: 5.806730700016487 seconds; ratio about 4.166x.

The executed command was:

```powershell
& $taskPython evaluate.py --checkpoint runs\final-frozen-wd06-refined-bigram\checkpoint.pt --device cpu --precision fp32 --threads 4 --split test --output runs\final-frozen-wd06-refined-bigram\test_cpu_fp32.json
```

Report `1.476789395067789` as the complete-test BPB. The predictor is now final;
do not use this result to make any subsequent model or parameter change.
