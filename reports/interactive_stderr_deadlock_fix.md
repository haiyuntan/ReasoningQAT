# Lean Interactive 管道阻塞修复（2026-10-17）

旧批次 `runs/realprover_autorun_v4` 的正式 P0 在题目 247 的第 24 次尝试等待 Lean 返回。
2026-10-17 00:38 UTC 诊断：Python worker 187074 等待 `pipe_read`；Lean 199253 等待
`pipe_write`。stderr 队列 65,419 / 65,536 字节，stdout 队列为零；10 秒采样 Lean
CPU/I/O 无推进。这是未消费 stderr 的管道反压阻塞，不能解释为复杂 tactic 计算。

`Interactive` 现在将 stderr 继承到父进程实验日志，stdout 继续独立承载 JSON 协议。
使用非阻塞 stdout、selector 和持久行缓冲，避免分片响应或一次多行造成丢失。
单次响应的墙钟上限为 1200 秒；超时或不完整 EOF 会清理独立进程组并显式报错，
不会静默当成 tactic 失败、跳题或成功。这个上限是运行保护的新增工程设置，
不声称来自论文；搜索预算、heartbeats、生成参数和固定题集不变。

旧批次进程树与孤儿 Lean 已停止，GPU 释放。根目录 GPTQ 日志及孤立 Lean 输入
移入 `runs/project_cleanup_20261017/`，没有删除旧 checkpoint、回执、原始结果或依赖缓存。
旧批次标记 `INTERRUPTED_FOR_REPAIR`，不合并进新批次的正式分数。

用户明确要求“修复后清理然后不用再次冒烟直接重启正式实验”。因此新增显式
`--formal-only --smoke-waiver <授权文件>` 入口；授权文件绑定新批次绝对路径、
当前源码/配置指纹、用户原文和通过的针对性验证报告。默认入口仍要求七项冒烟。
新入口只豁免冒烟，不豁免正式方法之间依赖、协议冻结、资产身份、单 GPU 锁、
导出检查、固定分母、回执及前后指纹一致性检查。没有伪造新代码的冒烟 PASS。

验证包括：2 MB stderr 洪泛后正常响应、连续缓冲响应、超时杀死启动器及后代、
EOF 报错、授权的批次/指纹绑定、默认门禁保留、控制器正式单阶段与失败即停，
以及真实 Lean 有效/无效 tactic、目标关闭和 commit 检查。
机器可读验证见 `runs/project_cleanup_20261017/validation.json`。

直接重启命令：

```bash
nohup env/bootstrap/bin/python -u research/scripts/run_smoke_then_formal.py \
  --results runs/realprover_formal_restart_v5 \
  --formal-only --smoke-waiver runs/realprover_formal_restart_v5/smoke_waiver.json \
  > runs/realprover_formal_restart_v5.nohup.log 2>&1 &
```

新正式批次从 P0 开始，随后按 P1–P6 顺序，后台持续运行。具体启动状态以新批次
`control/status.json` 为准。
