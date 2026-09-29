# 224x9 validation-only experiment

This controlled experiment changes only `depth` from 8 to 9. Width 224,
7 attention heads (head dimension 32), seed, training targets, optimizer,
matrix weight decay 0.5, MTP, EMA, and validation selection remain unchanged.
The existing 224x8 runs are not overwritten.

All development commands use `dev_data.py` or `evaluate_validation.py`; they
cannot open the test corpus. Do not use `evaluate.py --split test` while this
experiment or any other model development continues.

Run from `code/` in PowerShell. The first two lines use the working Python
runtime and the already installed packages in this workspace:

```powershell
$taskPython = 'C:\Users\YKB\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$env:PYTHONPATH = (Resolve-Path '.venv\Lib\site-packages').Path

& $taskPython -m unittest discover -s tests -v

& $taskPython train.py --implementation student --config configs\student_224x9_mtp2.json --device cuda --precision bf16 --seed 17 --steps 24000 --batch-size 32 --sampling random --learning-rate 0.0006 --warmup-steps 200 --min-lr-ratio 0.05 --weight-decay 0.5 --weight-decay-policy matrix --adam-beta2 0.95 --ema-decay 0.999 --ema-start-step 8000 --eval-every 500 --keep-best --run-dir runs\student-224x9-mtp2-ema-matrixwd05-s24000

& $taskPython evaluate_validation.py --checkpoint runs\student-224x9-mtp2-ema-matrixwd05-s24000\checkpoint.pt --device cpu --precision fp32 --threads 4

& $taskPython calibrate_checkpoint.py --checkpoint runs\student-224x9-mtp2-ema-matrixwd05-s24000\checkpoint.pt --device cuda --threads 4 --temperatures 1.04 1.08 1.12 1.16 --lambdas 0 0.04 0.07 0.10 --thetas 5 10 20 --run-dir runs\student-224x9-mtp2-ema-matrixwd05-s24000-calibrated

& $taskPython tune_bigram.py --checkpoint runs\student-224x9-mtp2-ema-matrixwd05-s24000-calibrated\checkpoint.pt --threads 4 --output runs\student-224x9-mtp2-ema-matrixwd05-s24000-calibrated\bigram_search.json

$search = Get-Content -Raw 'runs\student-224x9-mtp2-ema-matrixwd05-s24000-calibrated\bigram_search.json' | ConvertFrom-Json
& $taskPython export_bigram.py --checkpoint runs\student-224x9-mtp2-ema-matrixwd05-s24000-calibrated\checkpoint.pt --alpha $search.selected.alpha --lambda $search.selected.lambda --threads 4 --run-dir runs\student-224x9-mtp2-wd05-s24000-calibrated-bigram-selected

& $taskPython evaluate_validation.py --checkpoint runs\student-224x9-mtp2-wd05-s24000-calibrated-bigram-selected\checkpoint.pt --device cpu --precision fp32 --threads 4
```

Compare the final validation BPB with the current 224x8 candidate
`1.464390297786845`. Keep 224x9 only if it is lower and its paired validation
timing remains safely below the 5x baseline limit. Freeze the complete method
before any official test evaluation.

## Result — rejected

The run completed with validation selecting the step-18,000 EMA checkpoint.
The neural/cache predictor scored `1.4665910014063803` BPB on the locked
validation workflow. Repeating the same calibration grid kept temperature 1.12,
cache weight 0.07, and theta 10. The train-only bigram search selected alpha 1
and weight 0.02, with predicted validation BPB `1.4650307960341111`.

This is worse than the 224x8 complete candidate `1.464390297786845` by about
`0.0006405` BPB. The measured neural/cache validation scoring time was 30.503
seconds; compared with the recorded 4.921-second development baseline, this is
about 6.2x and therefore also fails the 5x timing requirement in that measurement.
Do not promote or test this checkpoint. Preserve it as evidence that adding the
ninth block increased compute without improving validation quality.
