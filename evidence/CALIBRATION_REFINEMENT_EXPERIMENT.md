# WD 0.6 local calibration refinement

The previous complete WD 0.6 candidate at `1.4612034232171305` validation BPB
was preserved unchanged. A new `refined-v1` directory searched the predeclared
local validation grid without loading test.

## Selected settings and result

- Temperature: 1.11.
- Within-window cache weight: 0.05.
- Cache theta: 12.
- Calibrated neural/cache BPB: `1.4616408485783534`.
- Train-only bigram alpha: 1.
- Bigram weight: 0.02.
- Exported CPU FP32 validation BPB: `1.4598314480072423`.
- Checkpoint SHA256: `c8c30df36e1e0b29beb11c36155d3f36020d2f8ad8462ee55d89e25a77db574f`.
- Inference assets: 36.305 MiB.
- Validation scoring time: 21.499 seconds, about 4.37x the recorded development
  baseline of 4.921 seconds.

The first evaluation attempt failed because the export directory did not exist:
PowerShell variables holding the selected bigram settings had not persisted into
the later terminal context. Export was rerun with the saved explicit settings
`--alpha 1 --lambda 0.02`, then CPU FP32 validation reproduced the search result.
The old best and every earlier experiment remain preserved. Do not access test
while development continues.
