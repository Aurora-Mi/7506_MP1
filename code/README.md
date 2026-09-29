# MP1 final code, checkpoint and reproduction instructions

## Frozen result

The submitted predictor was selected using validation and is frozen. Do not tune
or change it using the test result.

- Validation CPU FP32 BPB: **1.4598314480072423**
- Full-test CPU FP32 BPB: **1.476789395067789**
- Checkpoint: `runs/final-frozen-wd06-refined-bigram/checkpoint.pt`
- Checkpoint SHA256: `c8c30df36e1e0b29beb11c36155d3f36020d2f8ad8462ee55d89e25a77db574f`
- Final implementation: `student_bigram.py`, which depends on `student.py`

The final model has width 224, depth 8, seven attention heads, RoPE, RMSNorm,
SwiGLU and tied embeddings. Training used matrix-only weight decay 0.6, two
training-only MTP heads with weight 0.2, and the step-20,000 EMA weights. Final
inference uses temperature 1.11, a causal within-window cache with weight 0.05
and theta 12, and a train-derived bigram distribution with alpha 1 and weight
0.02. MTP heads are intentionally absent from the inference checkpoint.

## Installation

Use Python 3.12. From this `code/` directory on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.7.1 `
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-eval.txt
```

For training on a compatible NVIDIA GPU, install the matching PyTorch CUDA build
instead of the CPU build, then install `requirements.txt`.

## Verification

```powershell
python -m unittest discover -s tests -v
Get-FileHash runs\final-frozen-wd06-refined-bigram\checkpoint.pt `
  -Algorithm SHA256
```

The test suite contains 26 tests. The expected checkpoint hash is shown above.

## Validation reproduction

This command is safe for development because `evaluate_validation.py` is locked
to the supplied validation split:

```powershell
python evaluate_validation.py `
  --checkpoint runs\final-frozen-wd06-refined-bigram\checkpoint.pt `
  --device cpu --precision fp32 --threads 4 `
  --output reproduced_validation.json
```

Expected BPB: `1.4598314480072423`.

## Full-test reproduction

The method is already frozen; this command only reproduces the reported result:

```powershell
python evaluate.py `
  --checkpoint runs\final-frozen-wd06-refined-bigram\checkpoint.pt `
  --device cpu --precision fp32 --threads 4 --split test `
  --output reproduced_test.json
```

Expected full-test BPB: `1.476789395067789`, with 428,405 targets and
1,292,013 UTF-8 bytes. Small timing differences across machines are expected.

## Training command

The neural model was trained from random initialization using:

```powershell
python train.py --implementation student `
  --config configs\student_224x8_mtp2.json `
  --device cuda --precision bf16 --threads 4 --seed 17 `
  --steps 24000 --batch-size 32 --sampling random `
  --learning-rate 0.0006 --warmup-steps 200 `
  --min-lr-ratio 0.05 --weight-decay 0.6 `
  --weight-decay-policy matrix --adam-beta2 0.95 `
  --ema-decay 0.999 --ema-start-step 8000 `
  --eval-every 500 --keep-best --run-dir runs\repro-wd06
```

The recorded run selected the step-20,000 EMA checkpoint. A fresh run can differ
because GPU kernels are not guaranteed to be byte-identical.

Validation-only calibration used the following grid:

```powershell
python calibrate_checkpoint.py `
  --checkpoint runs\repro-wd06\checkpoint.pt `
  --run-dir runs\repro-wd06-calibrated `
  --device cuda --threads 4 `
  --temperatures 1.10 1.11 1.12 1.13 1.14 `
  --lambdas 0.02 0.03 0.04 0.05 0.06 `
  --thetas 12 16 20 24 28
```

The selected settings were temperature 1.11, cache weight 0.05 and theta 12.
The train-only bigram table was then exported as follows:

```powershell
python tune_bigram.py `
  --checkpoint runs\repro-wd06-calibrated\checkpoint.pt `
  --threads 4 --output runs\repro-wd06-calibrated\bigram_search.json
python export_bigram.py `
  --checkpoint runs\repro-wd06-calibrated\checkpoint.pt `
  --alpha 1 --lambda 0.02 --threads 4 `
  --run-dir runs\repro-wd06-bigram
```

The supplied frozen checkpoint should be used for evaluation; retraining is not
required.

## Evidence and provenance

`runs/final-frozen-wd06-refined-bigram/` contains the checkpoint, matching source
snapshots, training metrics, calibration and bigram records, validation and test
outputs, data hashes, resource measurements and the frozen manifest. The report
and the compact files under `../evidence/` document comparisons, ablations and
recorded search costs.

Only the supplied WikiText-2 training text was used to learn model weights and the
bigram table. Validation was used for selection. The tokenizer, data and official
scorer are unchanged.

## AI assistance disclosure

To some extent, AI helped me confirm the feasibility of the proposed changes and provided some guidance on aspects such as architectural design and parameter tuning.

See `../GUIDE.md` for the coursework requirements.
