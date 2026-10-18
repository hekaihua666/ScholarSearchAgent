# REG-MEQ v20 实验报告

## 冻结协议

- calibration40 确定 MEQ 参数，dev20 检查不退化，随后在 50 条独立盲测上一次性汇总。
- 训练 500、校准 150、calibration40、dev20、旧 holdout20 和预留 holdout30 均无查询重叠。
- 盲测检索 LIVE/FROZEN replay：50/50 一致。

## 核心数据

| 数据集 | 方法 | TP | Recall | F1 |
|---|---|---:|---:|---:|
| calibration40 | Baseline | 23 | 0.3051 | 0.0511 |
| calibration40 | REG | 28 | 0.3628 | 0.0620 |
| calibration40 | REG-MEQ | 29 | 0.3753 | 0.0643 |
| dev20 | Baseline | 22 | 0.4346 | 0.0933 |
| dev20 | REG | 25 | 0.5180 | 0.1069 |
| dev20 | REG-MEQ | 25 | 0.5180 | 0.1069 |
| blind50 | Baseline | 36 | 0.4070 | 0.0642 |
| blind50 | REG | 37 | 0.4170 | 0.0661 |
| blind50 | **REG-MEQ** | **38** | **0.4370** | **0.0680** |

盲测候选池共命中 51 篇，Recall=0.5357；REG-MEQ 将 Top20 命中从
36 提升到 38，全程未挤出原有相关论文。相对原 Top20，Recall 提高
7.37%，F1 提高 5.79%。

## 成本

- OpenAlex：348 次，6.96 次/查询。
- 已有本地交叉编码器评分：约 1.46 秒/查询。
- MEQ 路由计算：calibration/dev/blind50 分别 1.226/0.542/1.354 秒。
- 新增 API、LLM、Token：0。
