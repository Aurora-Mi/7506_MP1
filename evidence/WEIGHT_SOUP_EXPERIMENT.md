# WD 0.5 / WD 0.6 weight interpolation — rejected

This validation-only experiment interpolated aligned 224x8 checkpoint tensors.
Checkpoint A was the WD 0.5 validation-selected EMA model and checkpoint B was
the WD 0.6 validation-selected EMA model. The predeclared B weights were
0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, and 1. Test was not loaded.

Validation selected `alpha_a=0`, `alpha_b=1`, i.e. the unchanged WD 0.6 model,
at `1.4632018468867478` BPB. Every non-endpoint interpolation was worse; the
50/50 interpolation scored `1.5226238060386372`. The exported soup checkpoint's
CPU FP32 validation score, `1.4632018225527017`, differs from the original WD 0.6
measurement only by normal numerical variation.

Conclusion: do not promote or further calibrate the soup. Continue using the
already calibrated and train-bigram-enhanced WD 0.6 candidate at
`1.4612034232171305` validation BPB. This negative result is useful evidence that
the two weight-decay trajectories do not share a low-loss linear connection at
their independently validation-selected EMA steps.
