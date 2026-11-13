# tools

- `make_splits.py`：按 `configs/data_manifest.yaml` 的种子把样本划入 `data/splits/`。划分文件只含 `sample_id`。
- `summarize.py`：从 `runs/<run_id>/logs/` 和预登记写出 `artifacts/`。预登记缺失、划分重叠或 `protocol_version` 不一致时，退出码非 0，并且不写 `artifacts/pareto_tasks.csv`。

本目录的脚本是正式表的唯一写入者。
