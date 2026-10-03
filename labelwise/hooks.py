"""测试用同步钩子：在计算线程的关键位置回调，便于复现交错。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class Hooks:
    # 训练完成后、打分前回调：(project_id, train_ids)
    after_train: Optional[Callable[[int, list], None]] = None
    # 打分完成后、发布前回调：(project_id, selected_ids)
    after_score: Optional[Callable[[int, list], None]] = None
