# handoff

P4 的交接文件，名字固定为实验指导第 21 节：

- `configs_for_read_path.csv` 给课题二
- `feature_cost.csv` 给课题 W
- `priors.json`、`rules_s3.yaml` 给课题三；`screening.csv` 与 `pareto_tasks.csv` 使用 `artifacts/` 根目录里的正式副本，不在这里再存一份
- 面板 ILP 的解和最优性间隙与上面的文件一起放在本目录

每个文件的表头写明 `protocol_version`、模型修订和贪心全缓存这一协议。
