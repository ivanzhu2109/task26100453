# LabelWise 并发轮次推进库

内存态的标注轮次推进服务：标注批次闭合后自动训练并按预测熵选出下一批，
支持软删除与并发交错下的正确推进。真实使用 scikit-learn 的
`LogisticRegression(max_iter=1000)` 训练，全部状态保存在进程内存中。

## 运行测试

```bash
pip install -r requirements.txt   # 或: pip install scikit-learn numpy pytest
python3 -m pytest tests -v
```

## API

```python
from labelwise import LabelWise, Hooks

lw = LabelWise()
pid = lw.create(seed, pool, k)        # seed=[(id, feats, label)], pool=[(id, feats)] 或 dict
lw.submit(pid, round, sample, label)  # 同轮同样本同标注重试幂等；不同标注抛 LabelConflict
lw.remove(pid, sample)                # 软删除，幂等；种子不可删
snap = lw.snapshot(pid)               # 一致视图：轮次/阶段/批次/已收标注/模型版本及训练集ID
```

- `k` 为批量大小，取值 10–200；选样按预测熵降序、同分按 ID 升序，不足 k 取全部。
- 批次内所有未删除成员都有标注时自动闭合：无需调用方再发请求即进入训练。
- 训练集 = 种子 + 所有已接受且未删除的标注；模型版本与下一批一起发布，轮次只 +1。
- 无剩余候选时发布最终模型并进入 `exhausted`。

## 并发设计

- 项目锁只保护内存状态的短临界区；训练与整池打分在项目专属后台线程执行，
  `submit`/`remove`/`snapshot` 从不等待计算结束，项目之间互不阻塞。
- 每次删除递增 `_delete_version`；计算启动时快照该版本，发布前比对，
  不一致则作废旧结果并用最新状态重算——删除返回之后的任何发布都不含已删样本，
  删除停止后必然最终推进。
- 测试钩子 `Hooks(after_train, after_score)` 在计算线程的关键位置回调，
  测试用 `threading.Event` 在这些同步点暂停/放行以确定性地复现交错。
