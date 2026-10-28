# 配额清理后的正式评估续跑

从已停止的v10创建v12，保持v10模型、原始attempt、遥测与回执不可变。允许源状态STOPPED_DISK_QUOTA，并检查源控制器和评估进程确实不再运行。恢复既有本地Lean路径、同一miniF2F244题与pass64搜索预算、vLLM0.85；不训练、不重新量化。

检索资产路径已经迁移。本次先比较旧Git提交中的检索清单和当前清单，要求service source revision以及全部(size,SHA256)多重集合相同；路径变化本身不改变模型/索引内容身份。模型复用仍要求全部准备产物哈希及训练/量化代码不变。

冻结协议中的retrieval_assets.json清单哈希同步更新到迁移后路径清单，保留前后哈希、内容身份一致性及真实检索健康验证来源。所有其余冻结资产哈希原样保留。初始化阶段发现旧绑定后停止了首轮模型审计，未开始新推理；清除仅包含本次硬链接/元数据的未完成导入后，从同一不可变v10重新初始化。

所有已提交attempt仍经过严格JSON、槽位范围、theorem statement及完整Lean成功标记验证，复制后校验内容SHA。占用槽不覆盖，成功题或64槽耗尽题不再搜索；未发布.tmp只登记来源，不当成完成尝试。损坏的已发布JSON拒绝启动，不能静默丢弃后补预算。历史timeout和基础设施错误记录保留；新批次遥测重新生成。旧RNG内存状态未保存，因此不保证与不中断运行逐bit等价。

P1/P2若已有完整244题结果，本次额外审计原PASS回执及每个证据哈希，核对旧聚合与新导入原始证据完全一致，使用正常方法检查/receipt writer生成绑定新路径的新证据回执，并显式标记`inherited_completed_no_new_inference`。原preflight与retrieval identity作为历史证据保留，不冒称在新批次重新推理或执行了新的P1/P2评估。控制器直接将这些已完成方法置DONE，不为其启动vLLM。

GPU3共享检索，GPU0/1/2先续跑P3/P4/P5，任一卡结束后领取P6。保持单方法失败隔离。当前授权沿用不重复全套smoke的正式续跑流程，绑定本次针对性测试报告，不生成新的全套smoke PASS。

v11首次启动暴露了共享服务健康检查饥饿：单线程Flask在处理慢检索时，其他客户端的/health超过5秒并直接退出。P3/P4失败不影响P5/P6，但修复前受控停止了全批次；没有新增已提交attempt，因此v12仍可直接从不可变v10导入同一起点。修复采用threaded HTTP服务，仅在实际model/FAISS retrieve周围持互斥锁，health不需要GPU锁；仍禁止并发GPU检索，检索数学/返回值和正式采样预算不变。shared ready允许临时连接/健康检查超时重试至总deadline，客户端不会接管服务生命周期。真实HTTP回归执行生产server.__main__，慢检索6秒时health仍在2秒内返回，两个检索严格按start/end/start/end执行。

```bash
env/bootstrap/bin/python research/scripts/launch_evaluation_resume.py \
  --source runs/realprover_formal_four_gpu_v10 \
  --results runs/realprover_formal_four_gpu_v12 \
  --validation-report runs/evaluation_resume_20261028/validation.json
```

初始化先审计大模型和原始结果，再nohup控制器。v12初始化日志位于`runs/evaluation_resume_20261028/initializer_v12.nohup.log`；正式控制器日志为`runs/realprover_formal_four_gpu_v12.nohup.log`。启动前完整进度保存到新批次`control/previous_progress.json`；已完成继承的审计记录位于`control/completed_inheritance.json`。v11日志和失败状态保留以供审计。
