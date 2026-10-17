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

## 后续：超时消耗一次搜索尝试，而非终止整批

用户进一步要求：出现 Lean 等待超时时，本次按错误计算并继续 pass@64 的剩余尝试。
据此增加独立 `InteractiveResponseTimeout`，并在批量搜索边界识别直接抛出及
`SearchError.error_type` 包装的这类超时。保存 `success=false`、超时操作和1200秒阈值，
标记 `pipeline_attempt.status=timeout` 与 `consumes_attempt=true`；即使模型调用列表为空，
也消耗一次预算。下次尝试使用新的 Lean 会话，不增加额外重试。若剩余尝试成功，
该题仍算成功；64次都失败才计该题错误。报告新增 `lean_timeout_attempts`。

任意退出路径均关闭 Lean 进程组并删除临时输入。其他搜索/通信/环境异常重新抛出，
worker 非零退出由主进程上报，不混入模型错误率。原先批量循环实际会捕获异常继续，
但没有保存对应失败尝试；本次补全计数和证据，避免该统计缺口。

新策略验证20项：真实批量循环的超时后成功、64次超时耗尽、包装超时、基础设施
错误传播，汇总去重与超时次数，以及原管道/授权/门禁/控制器检查全部通过。
`runs/project_cleanup_20261017/timeout_attempt_tests.json` 保存完整测试输出。
上一轮真实 Lean 合法/错误 tactic、目标关闭与 commit 验证保持适用。

v5 已标记 `INTERRUPTED_FOR_TIMEOUT_ACCOUNTING`。最终批次改用
`runs/realprover_formal_restart_v6`，使用新指纹绑定的用户授权直接正式启动；
不把v5或历史冒烟回执改成新代码的 PASS，不混合旧分数。

## Lean 递归崩溃恢复与队列退出修复

v6 在 2026-10-17 09:54 左右因题目247一次 `native_decide` 相关计算引发
`Nat.factorial` 深递归，Lean stderr 记录 `deep recursion was detected at interpreter`。
stdout EOF 原本抛 `RuntimeError`，被搜索器当作普通无效 tactic 吞掉，随后继续访问
已关闭会话，引发 closed-file 错误。worker退出后，主进程的 multiprocessing.Queue
还有未消费题目，解释器等待队列 feeder 冲刷，导致控制状态仍停在运行中。

新增 `InteractiveSessionError(Exception)`，明确区别于无效 tactic 的 RuntimeError。
Lean EOF/崩溃、关闭会话写入及非法JSON均结束会话并抛出此类型。批量边界识别直接或
SearchError 包装的会话错误，保存 `status=lean_error`、操作、退出码（可获得时）和
失败结果，消耗一次尝试，再用全新会话执行下一次。超时仍按原独立类型计数。
汇总增加 `lean_error_attempts`；最终每题最多64次、成功按题去重、分母244不变。

批量运行通过 finally 终止遗留worker、取消队列feeder退出等待、关闭队列并shutdown
manager。仍然无法恢复的模型/服务故障显式失败退出，不再因无人消费任务队列而挂住。

25项针对性验证覆盖真实管道子进程异常退出、异常不被误当无效tactic、非法JSON、
直接/包装Lean崩溃后下一次成功、64次崩溃耗尽预算、超时保留、汇总及原门禁。
另用真实multiprocessing子进程模拟worker失败和10MB未消费任务队列，确认进程在
15秒测试期限内退出。证据：`runs/project_cleanup_20261017/lean_crash_tests.json`。

v6 已停止且保留原始结果；新批次 `runs/realprover_formal_restart_v7` 继续用户授权的
直接正式启动策略，不重跑完整冒烟，不把旧回执冒充当前代码的PASS。
