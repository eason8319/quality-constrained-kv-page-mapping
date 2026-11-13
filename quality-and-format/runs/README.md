# runs

除 `_template` 外，每个子目录是一次运行。运行名规则见[实验目录](../实验目录.md)。

开始运行时复制[预登记模板](_template/preregistration.yaml)。P0/P1 保存当期设置和实测记录。P2 按实验指导第 15 节填实冻结项并提交，P3 引用该快照。`logs/` 和 `cache/` 不进 git。

不同 `protocol_version` 的运行并列保留，不互相覆盖。
