# splits

这里只放划分清单，每行一个 `sample_id`，按任务族和 `D_cal`、`D_sel`、`D_test` 分文件。文件由 `tools/make_splits.py` 生成。

P2 提交之后这些文件只读。原文、token 和下载缓存放在 `data/raw/`，该目录被 git 忽略。
