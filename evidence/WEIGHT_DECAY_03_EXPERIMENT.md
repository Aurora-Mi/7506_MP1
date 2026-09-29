# 下一轮：224×8，矩阵 weight decay 0.3

只将上一轮训练的 `--weight-decay 0.2` 改为 `0.3`，使用相同架构配置，
从随机初始化重新训练。训练时温度仍为 1.12，与以前实验相同；之后再校准。
不修改模型结构，不从带 bigram 的 checkpoint 续训，不覆盖现有结果。

在 code/ 目录、激活虚拟环境后运行：

```powershell
python train.py --implementation student --config configs/student_224x8_mtp2.json --device cuda --precision bf16 --seed 17 --steps 24000 --batch-size 32 --sampling random --learning-rate 0.0006 --warmup-steps 200 --min-lr-ratio 0.05 --weight-decay 0.3 --weight-decay-policy matrix --adam-beta2 0.95 --ema-decay 0.999 --ema-start-step 8000 --eval-every 500 --keep-best --run-dir runs/student-224x8-mtp2-ema-matrixwd03-s24000
python evaluate.py --checkpoint runs/student-224x8-mtp2-ema-matrixwd03-s24000/checkpoint.pt --device cpu --precision fp32 --threads 4 --split validation
```

第一阶段基础模型的同等对照分数为 **1.4771008840560214**，不是加入
bigram 后的 1.475067276886635。结果出来后，再做同等参数搜索：

```powershell
python calibrate_checkpoint.py --checkpoint runs/student-224x8-mtp2-ema-matrixwd03-s24000/checkpoint.pt --device cuda --threads 4 --temperatures 1.04 1.08 1.12 1.16 --lambdas 0 0.04 0.07 0.10 --thetas 5 10 20 --run-dir runs/student-224x8-mtp2-ema-matrixwd03-s24000-calibrated
python evaluate.py --checkpoint runs/student-224x8-mtp2-ema-matrixwd03-s24000-calibrated/checkpoint.pt --device cpu --precision fp32 --threads 4 --split validation
python tune_bigram.py --checkpoint runs/student-224x8-mtp2-ema-matrixwd03-s24000-calibrated/checkpoint.pt --threads 4 --output runs/student-224x8-mtp2-ema-matrixwd03-s24000-calibrated/bigram_search.json
```

Bigram 的实际 alpha/lambda 根据新搜索结果确定，不能预设旧值仍然最优。
导出并用官方 CPU FP32 评分器复核后，与完整预测器基准 1.475067276886635 比较。
仅在 validation 调参，方法冻结后才测 test，确认完整 CPU 时间和实际 RAM 限制。
AI assistance: Codex helped prepare this controlled experiment and preserve prior results.
