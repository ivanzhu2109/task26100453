# labelwise

LabelWise 并发轮次推进库：标注批次闭合、训练选样与软删除。
状态全部在内存中，面向数据可装入内存、进程不中途退出的场景。

## 安装与测试

```bash
pip install -r requirements.txt
python -m pytest tests -q
```

## 用法

```python
from labelwise import LabelWise

lw = LabelWise()
seed = [(1, [0.1, 0.2], "cat"), (2, [0.9, 0.8], "dog"), ...]  # (id, 特征, 标签)，至少两类
pool = [(101, [0.3, 0.1]), (102, [0.8, 0.7]), ...]           # (id, 特征)
pid = lw.create(seed, pool, k=10)   # 种子训练 + 发布第一批，k ∈ [10, 200]

lw.submit(pid, 1, 101, "cat")       # 提交第 1 轮标注；相同标注重试幂等，不同标注拒绝
lw.remove(pid, 102)                 # 软删除（含已标注样本），返回即生效，重复删除无副作用
snap = lw.snapshot(pid)             # 一致视图：轮次/阶段/当前批次/已接收标注/模型版本及训练集
```

当前批次所有未删除成员都有标注时自动进入训练（真实 `LogisticRegression`，固定参数），
训练与整池打分完成后，模型版本与下一批在同一次发布中生效；无剩余候选时发布最终模型
并进入 `exhausted`。选样按预测熵降序，同分按 ID 升序，不足 k 条取剩余全部。

## 并发设计

- 项目状态只在项目锁下读写；训练与整池打分在独立计算线程中执行、不持锁，
  `submit`/`remove`/`snapshot` 从不等待计算。
- 每个项目独立的计算线程，项目间不串行化。
- 计算开始时对输入做一致性快照（`state_version`）；发布前在锁内校验。
  任何删除都会使旧快照失效：旧计算结果作废、重取输入重算，
  因此删除返回之后的发布绝不含已删除样本；删除停止后必然推进。
- 测试通过 `labelwise.Hooks` 的同步点（`train_start`/`score_done`/`published`）
  复现交错，不依赖 sleep。
