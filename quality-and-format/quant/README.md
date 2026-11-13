# quant

黄金量化，语义见实验指导第 6 节。

- `golden.py`：分组、舍入、常量组、写回 bfloat16。
- `test_golden.py`：第 6.3 节的按位单测。

单测输出以后写到对应运行的 `runs/<run_id>/`，正式哈希由汇总抄入 `artifacts/`。这里不放日志。
