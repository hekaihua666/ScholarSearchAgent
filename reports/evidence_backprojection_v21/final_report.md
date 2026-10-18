# EBP v21 冻结实验报告

## 主要结果

| 数据集 | 原候选池 TP / Recall | EBP 后 TP / Recall | 新增命中 |
|---|---:|---:|---:|
| calibration40 | 35 / 0.4226 | 44 / 0.5185 | +9 |
| dev20 | 26 / 0.5346 | 28 / 0.5721 | +2 |
| blind50 | 50 / 0.5323 | **53 / 0.5623** | **+3** |

Blind50 中新增命中的论文来自 3 个查询：

- `AutoScholarQuery_test_389`：Learning Transferable Visual Models From Natural Language Supervision
- `AutoScholarQuery_test_537`：Language Models are Few-Shot Learners
- `AutoScholarQuery_test_716`：Provably Efficient Maximum Entropy Exploration

在相同冻结输入上，EBP 将 Blind50 候选召回从 0.5323 提高到 0.5623，相对提高 5.64%。calibration40、dev20 与 Blind50 均出现正向候选增益。

## 预算—召回曲线

Blind50 的目标预算从 10 增加到 75 时，候选 TP 依次达到 51, 51, 52, 53；100 个目标保持 TP=53。冻结配置采用 100，以优先保留 calibration40 上最后一篇新增论文，并为后续排序器提供完整证据边界。

## 最终集合

本轮只扩展候选证据边界，不强制把新增论文写入 Top20。最终集合继续使用已冻结的 REG-MEQ 路由，Blind50 保持 TP=38、Recall=0.4370、F1=0.0680。这使召回扩展与最终集合决策相互解耦：EBP 负责找回可达论文，REG-MEQ 负责维持当前排序稳定性。

## 推理成本

- OpenAlex：每个查询 1 次批量请求；Blind50 共 50 次。
- Blind50 平均批量检索耗时：2.429 秒/查询。
- Blind50 平均本地交叉编码与关系特征计算：0.613 秒/查询。
- 新增 LLM 调用与 Token：0。

## 结论

EBP 证明当前系统的部分召回缺口并非不可达：已有候选的引文元数据中包含可以恢复的有效论文标识。通过全候选多视图证据聚合和一次批量物化，系统在三个隔离数据集上都扩大了候选召回边界。下一轮排序优化将直接面向这些新增候选，训练能够识别“强图证据、弱词面重合”论文的选择器。
