# pack

页布局和字节模型，语义见实验指导第 7 节。

- `layout.py`：描述符、K/V 分段、突发对齐、\(B_{\mathrm{DRAM}}\) 与 \(b_{\mathrm{eff}}\)。
- `test_layout.py`：第 7.3 节的打包往返和手工例子。

`artifacts/byte_model.json` 由工具写出，不放在本目录。
