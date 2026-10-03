"""项目注册表：对外提供 create/submit/remove/snapshot 四个操作。"""
from __future__ import annotations

import threading
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .errors import ProjectNotFound
from .hooks import Hooks
from .project import Project, Snapshot

PoolInput = Union[Mapping[int, Sequence[float]], Iterable[Tuple[int, Sequence[float]]]]
SeedInput = Iterable[Tuple[int, Sequence[float], Any]]

MIN_K = 10
MAX_K = 200


def _normalize_seed(seed: SeedInput) -> List[Tuple[int, Sequence[float], Any]]:
    return [(int(sid), list(feat), label) for sid, feat, label in seed]


def _normalize_pool(pool: PoolInput) -> List[Tuple[int, Sequence[float]]]:
    items = pool.items() if isinstance(pool, Mapping) else pool
    return [(int(pid), list(feat)) for pid, feat in items]


class LabelWise:
    """内存态 LabelWise 服务。所有状态保存在进程内存中。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()  # 仅保护注册表本身，不参与项目计算
        self._projects: Dict[int, Project] = {}
        self._next_id = 1

    def create(
        self,
        seed: SeedInput,
        pool: PoolInput,
        k: int,
        hooks: Optional[Hooks] = None,
    ) -> int:
        """创建项目，同步完成种子训练并发布第一批，返回项目 ID。"""
        if not isinstance(k, int) or not (MIN_K <= k <= MAX_K):
            raise ValueError(f"k must be an int in [{MIN_K}, {MAX_K}], got {k!r}")
        seed_items = _normalize_seed(seed)
        pool_items = _normalize_pool(pool)
        if not seed_items:
            raise ValueError("seed must not be empty")
        if len({label for _, _, label in seed_items}) < 2:
            raise ValueError("seed must contain at least two classes")
        ids = [sid for sid, _, _ in seed_items] + [pid for pid, _ in pool_items]
        if len(set(ids)) != len(ids):
            raise ValueError("sample ids must be unique")
        widths = {len(feat) for _, feat, _ in seed_items} | {
            len(feat) for _, feat in pool_items
        }
        if len(widths) > 1 or (widths and 0 in widths):
            raise ValueError("features must be non-empty and fixed-length")

        with self._lock:
            project_id = self._next_id
            self._next_id += 1
        # 种子训练可能耗时，放在注册表锁外；项目对象发布前对外不可见。
        project = Project(project_id, seed_items, pool_items, k, hooks)
        with self._lock:
            self._projects[project_id] = project
        return project_id

    def submit(self, project: int, round_no: int, sample: int, label: Any) -> None:
        self._get(project).submit(round_no, sample, label)

    def remove(self, project: int, sample: int) -> None:
        self._get(project).remove(sample)

    def snapshot(self, project: int) -> Snapshot:
        return self._get(project).snapshot()

    def _get(self, project_id: int) -> Project:
        with self._lock:
            try:
                return self._projects[project_id]
            except KeyError:
                raise ProjectNotFound(f"unknown project {project_id}") from None
