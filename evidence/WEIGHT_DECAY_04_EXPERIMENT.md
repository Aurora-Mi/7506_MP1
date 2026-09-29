# 下一轮：224×8，矩阵 weight decay 0.4

只将上一轮的矩阵 weight decay 0.3 改为 0.4，其他训练设置不变。
使用现有模型配置，从头训练；不加载旧 checkpoint，也不覆盖最佳结果。
训练时评分温度仍为 1.12，便于同等基础模型比较，之后再独立校准。
Trainer 已支持 weight decay 参数，不需要修改模型结构。

在已激活虚拟环境的 code/ 目录运行：

```powershell
python train.py --implementation student --config configs/student_224x8_mtp2.json --device cuda --precision bf16 --seed 17 --steps 24000 --batch-size 32 --sampling random --learning-rate 0.0006 --warmup-steps 200 --min-lr-ratio 0.05 --weight-decay 0.4 --weight-decay-policy matrix --adam-beta2 0.95 --ema-decay 0.999 --ema-start-step 8000 --eval-every 500 --keep-best --run-dir runs/student-224x8-mtp2-ema-matrixwd04-s24000
python evaluate.py --checkpoint runs/student-224x8-mtp2-ema-matrixwd04-s24000/checkpoint.pt --device cpu --precision fp32 --threads 4 --split validation
```

第一阶段与基础模型 1.4712857763278189 对照，不直接与包含 bigram 的
1.4689658222529052 比较。结果出来后，再执行同等校准和训练集 bigram 搜索：

```powershell
python calibrate_checkpoint.py --checkpoint runs/student-224x8-mtp2-ema-matrixwd04-s24000/checkpoint.pt --device cuda --threads 4 --temperatures 1.04 1.08 1.12 1.16 --lambdas 0 0.04 0.07 0.10 --thetas 5 10 20 --run-dir runs/student-224x8-mtp2-ema-matrixwd04-s24000-calibrated
python evaluate.py --checkpoint runs/student-224x8-mtp2-ema-matrixwd04-s24000-calibrated/checkpoint.pt --device cpu --precision fp32 --threads 4 --split validation
python tune_bigram.py --checkpoint runs/student-224x8-mtp2-ema-matrixwd04-s24000-calibrated/checkpoint.pt --threads 4 --output runs/student-224x8-mtp2-ema-matrixwd04-s24000-calibrated/bigram_search.json
```

随后按照新搜索选出的 alpha/lambda 导出并用官方 CPU FP32 评分器确认。
完整模型最终与 1.4689658222529052 比较；若没有提升，保留 WD 0.3 最佳模型。
所有选择只用 validation，方案冻结后才测 test。仍需确认完整 CPU 时间和实际 RAM。
AI assistance: Codex helped preserve prior results and prepare this controlled experiment.
