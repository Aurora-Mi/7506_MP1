# 下一轮：224×8 矩阵 weight decay 0.2

此实验只改变训练参数 `--weight-decay 0.2`。架构、seed、采样方式、处理
targets、学习率、MTP、EMA 和训练时验证设置均沿用原 224×8 实验。
无需修改模型实现；现有 trainer 支持该参数。不是从 bigram 模型续训。

从 code/ 目录运行，先激活虚拟环境：

```powershell
python train.py --implementation student --config configs/student_224x8_mtp2.json --device cuda --precision bf16 --seed 17 --steps 24000 --batch-size 32 --sampling random --learning-rate 0.0006 --warmup-steps 200 --min-lr-ratio 0.05 --weight-decay 0.2 --weight-decay-policy matrix --adam-beta2 0.95 --ema-decay 0.999 --ema-start-step 8000 --eval-every 500 --keep-best --run-dir runs/student-224x8-mtp2-ema-matrixwd02-s24000
python evaluate.py --checkpoint runs/student-224x8-mtp2-ema-matrixwd02-s24000/checkpoint.pt --device cpu --precision fp32 --threads 4 --split validation
```

基础模型的同等对照分数是 1.4830335787593925（温度 1.12，无 bigram）。
正式比较最终预测器时，再执行同等参数范围的校准和同一 bigram 组合：

```powershell
python calibrate_checkpoint.py --checkpoint runs/student-224x8-mtp2-ema-matrixwd02-s24000/checkpoint.pt --device cuda --threads 4 --temperatures 1.04 1.08 1.12 1.16 --lambdas 0 0.04 0.07 0.10 --thetas 5 10 20 --run-dir runs/student-224x8-mtp2-ema-matrixwd02-s24000-calibrated
python export_bigram.py --checkpoint runs/student-224x8-mtp2-ema-matrixwd02-s24000-calibrated/checkpoint.pt --alpha 20 --lambda 0.02 --threads 4 --run-dir runs/student-224x8-mtp2-wd02-s24000-calibrated-bigram-a20-w002
python evaluate.py --checkpoint runs/student-224x8-mtp2-wd02-s24000-calibrated-bigram-a20-w002/checkpoint.pt --device cpu --precision fp32 --threads 4 --split validation
```

完整预测器的对照分数是 1.4801953932045586。若 bigram 在新模型上变差，
保留校准后的纯神经/cache 模型，不强制使用插值。所有实验仅用 validation；
方案冻结后才做 test 评估，检查完整 CPU 时间和实际 RAM 限制。
AI assistance: Codex helped prepare this experiment recipe and preserve prior results.
