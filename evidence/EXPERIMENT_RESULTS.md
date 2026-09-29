# 当前已记录的最佳验证结果

最后更新：2026-09-27（Asia/Shanghai）。所有分数均为固定协议
`7506-mp1-wt2-v2` 的完整 validation、CPU FP32、4 线程结果。

| 实验 | Validation BPB | CPU 验证秒数 | 保存位置 |
|---|---:|---:|---|
| 192×8，24000 步，温度校准后 | 1.4878460648946534 | 18.182381400023587 | `runs/student-192x8-mtp2-ema-matrixwd-s24000-calibrated/` |
| 224×8，24000 步，校准前 | 1.4830335787593925 | 21.53075410006568 | `runs/student-224x8-mtp2-ema-matrixwd-s24000/` |
| 224×8，温度校准后 | 1.4823521516489688 | 21.266041100025177 | `runs/student-224x8-mtp2-ema-matrixwd-s24000-calibrated/` |
| 224×8，校准 + bigram，WD 0.1 | 1.4801953932045586 | 21.871359699987806 | `runs/student-224x8-mtp2-s24000-calibrated-bigram-a20-w002/` |
| 224×8，WD 0.2，校准前 | 1.4771008840560214 | 21.214449700084515 | `runs/student-224x8-mtp2-ema-matrixwd02-s24000/` |
| 224×8，WD 0.2，校准后 | 1.476997457999043 | 20.44932949997019 | `runs/student-224x8-mtp2-ema-matrixwd02-s24000-calibrated/` |
| 224×8，WD 0.2，校准 + bigram | 1.475067276886635 | 21.58291959995404 | `runs/student-224x8-mtp2-wd02-s24000-calibrated-bigram-a5-w002/` |
| 224×8，WD 0.3，校准前 | 1.4712857763278189 | 22.923116000019945 | `runs/student-224x8-mtp2-ema-matrixwd03-s24000/` |
| 224×8，WD 0.3，校准后（参数未变） | 1.4712857763278189 | 22.32071880006697 | `runs/student-224x8-mtp2-ema-matrixwd03-s24000-calibrated/` |
| **224×8，WD 0.3，校准 + bigram，当前最佳** | **1.4689658222529052** | **23.671487399959005** | `runs/student-224x8-mtp2-wd03-s24000-calibrated-bigram-a20-w002/` |

三个 224×8 WD 实验均训练 24000 步，处理 196608000 targets。
WD 0.1/0.2/0.3 分别选中第 17500/18000/19000 步 EMA。
校准和 bigram 导出没有新增神经训练；统计仅来自训练集。
当前最佳温度 1.12、cache lambda 0.07、theta 10、bigram alpha 20、权重 0.02。
相比此前 WD 0.2 完整模型降低 0.0061014546337298 BPB。

当前最佳 checkpoint SHA256：
`5ec068dfe4ddd61a27cdec53e4699aa7f11380878582c25f56328fca11be6045`。

当前独立备份：`runs/best-1p468966-224x8-wd03-bigram-backup/`，包含 checkpoint、
配置、训练/校准/导出/评估记录、逐窗口损失与对应代码；备份 checkpoint 哈希已核对一致。
此前所有备份继续保留；当前 bigram 模型推理资产合计约 36.31 MiB。

下一步：矩阵 WD 0.4 实验，使用独立目录
`runs/student-224x8-mtp2-ema-matrixwd04-s24000/`，不同时改变学习率、架构、
采样方式、seed、EMA 或训练步数。基础模型与 WD 0.3 同设置的
1.4712857763278189 比较；经同等校准和 bigram 搜索后才与完整最佳预测器
1.4689658222529052 比较。该实验尚未完成，不能保证提升。

后续实验使用新目录，不能覆盖这些最佳结果。
本记录不代表已验证最终 test BPB、CPU 全 test 时间限制或实际峰值 RAM。
