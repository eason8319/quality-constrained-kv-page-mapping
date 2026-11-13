# features

页关闭特征与嵌套预测器，定义见实验指导第 11 节和第 12.4 节。

- `extract.py`：按 `F0`、`F1`、`F2`、`F3`、`oracle` 分列写出。
- `fit_rules.py`：在 `D_cal` 上拟合 `M0`–`M3`，在 `D_sel` 上选择 \(\tau\) 和 `S3` 的那一条规则。

读取列名时，`oracle` 前缀应直接报错。拟合出的规则写入对应运行目录；冻结后的 `rules_s3.yaml` 由 P4 复制到 `artifacts/handoff/`。
