# eval

三种干预和任务评测。干预名只能是实验指导第 8 节列出的那些。

- `fork.py`：`recon`、`local_attn`、`closure_fork`、`closure_fork_joint`。
- `metrics.py`：重建误差、局部注意力、\(S\) 与 \(S_{\mathrm{key}}\)。
- `fp_gate.py`：全精度守门。
- `pilot_timing.py`：第 10.2 节计时表。
- `run_panel.py`：分布块和配对面板。
- `run_paired_pages.py`：同页配对。
- `run_allocations.py`：全缓存分配 `S0`–`S4`。

逐条记录写入 `runs/<run_id>/logs/`，不写进本目录。
