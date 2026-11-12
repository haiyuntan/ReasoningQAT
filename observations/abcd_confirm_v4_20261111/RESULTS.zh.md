# ABCD 扩大规模观察实验

状态：CONFIRMATION_DONE，失败阶段为零。

P0/P3/P4/P5/P6 各完成256个评分案例：128个新的 miniF2F-valid 题目和64组规则改名对（128实例）。P3/P4/P5/P6 各完成32题×28层重放，共3584条层级记录。主要比较是 P3/P4/P6；P5 是预注册的次要基线。

- 图表：[PNG](plots/ABCD_confirmation.png)、[PDF](plots/ABCD_confirmation.pdf)
- 数值汇总：[confirmation_summary.json](metrics/confirmation_summary.json)
- 分析报告：[results.md](reports/results.md)
- 预注册：[PREREGISTRATION.md](reports/PREREGISTRATION.md)
- 完成说明：[EXECUTION_STATUS.md](reports/EXECUTION_STATUS.md)

这批结果是观察实验，不是训练或 theorem-level pass@64 评估。A/B/C 使用预注册的配对/聚类 bootstrap；D 的全层曲线属于描述性分析。数据来源的语义重叠等限制见预注册和 README。

发布包包含结果、冻结输入、代码和 SHA256 校验；不包含模型权重、环境、激活缓存或 Lean 二进制。旧先导实验保留在相邻 abcd_20261111 目录。
