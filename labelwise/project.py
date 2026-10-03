"""单项目状态机：批次闭合、异步训练/打分、软删除、一致快照。

并发设计：
- 项目锁只保护内存状态的短临界区，训练与整池打分在锁外的后台线程执行。
- 每次删除递增 _delete_version；计算启动时快照该版本，发布前比对，
  不一致则作废并用最新状态重算，保证删除返回后的任何发布都不含已删样本。
- 每个项目同一时刻至多一个计算线程，项目之间完全独立。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from .errors import InvalidState, LabelConflict, SampleNotFound, StaleRound
from .hooks import Hooks

PHASE_LABELING = "labeling"
PHASE_TRAINING = "training"
PHASE_EXHAUSTED = "exhausted"


@dataclass(frozen=True)
class ModelInfo:
    version: int
    train_ids: Tuple[int, ...]


@dataclass(frozen=True)
class Snapshot:
    project_id: int
    round: int
    phase: str
    batch: Tuple[int, ...]
    received: Dict[int, Any]
    models: Tuple[ModelInfo, ...]
    deleted: Tuple[int, ...]
    batch_version: int  # 发布当前批次时一同发布的模型版本


@dataclass
class _ComputeInput:
    delete_version: int
    train_ids: List[int]
    train_x: np.ndarray
    train_y: np.ndarray
    cand_ids: List[int]
    cand_x: np.ndarray


def _train_model(x: np.ndarray, y: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000)
    model.fit(x, y)
    return model


def _rank_by_entropy(model: LogisticRegression, cand_ids: List[int], cand_x: np.ndarray) -> List[int]:
    """按预测熵降序、ID 升序返回候选 ID。"""
    if not cand_ids:
        return []
    proba = model.predict_proba(cand_x)
    log_p = np.log(np.clip(proba, 1e-300, None))
    entropy = -(proba * log_p).sum(axis=1)
    order = sorted(range(len(cand_ids)), key=lambda i: (-entropy[i], cand_ids[i]))
    return [cand_ids[i] for i in order]


class Project:
    def __init__(
        self,
        project_id: int,
        seed: List[Tuple[int, Sequence[float], Any]],
        pool: List[Tuple[int, Sequence[float]]],
        k: int,
        hooks: Optional[Hooks] = None,
    ) -> None:
        self.project_id = project_id
        self.k = k
        self._hooks = hooks or Hooks()
        self._lock = threading.Lock()

        self._seed_ids = [s[0] for s in seed]
        self._seed_x = np.asarray([s[1] for s in seed], dtype=float)
        self._seed_y = np.asarray([s[2] for s in seed])
        self._features: Dict[int, np.ndarray] = {
            pid: np.asarray(feat, dtype=float) for pid, feat in pool
        }

        self._labels: Dict[int, Any] = {}          # 已接受且未删除的标注（训练用）
        self._history: Dict[Tuple[int, int], Any] = {}  # (round, sample) -> label
        self._deleted: set = set()
        self._round = 0
        self._phase = PHASE_TRAINING
        self._batch: List[int] = []
        self._received: Dict[int, Any] = {}        # 当前轮次已接收的标注
        self._models: List[Tuple[ModelInfo, LogisticRegression]] = []
        self._batch_version = 0

        self._delete_version = 0
        self._computing = False

        # 种子训练与第一批发布在 create 内同步完成。
        seed_train = _train_model(self._seed_x, self._seed_y)
        first_batch = self._select(seed_train, self._candidates_locked())
        self._publish_locked(seed_train, list(self._seed_ids), first_batch)

    # ---------- 对外操作 ----------

    def submit(self, round_no: int, sample: int, label: Any) -> None:
        with self._lock:
            key = (round_no, sample)
            if key in self._history:
                if self._history[key] == label:
                    return  # 相同标注重试：幂等成功
                raise LabelConflict(
                    f"sample {sample} round {round_no} already labeled "
                    f"{self._history[key]!r}, got {label!r}"
                )
            if self._phase == PHASE_EXHAUSTED:
                raise InvalidState("project exhausted")
            if round_no != self._round or self._phase != PHASE_LABELING:
                raise StaleRound(
                    f"current round is {self._round} ({self._phase}), got {round_no}"
                )
            if sample in self._deleted:
                raise InvalidState(f"sample {sample} is deleted")
            if sample not in self._batch:
                raise SampleNotFound(f"sample {sample} not in current batch")
            self._history[key] = label
            self._labels[sample] = label
            self._received[sample] = label
            self._check_closure_locked()

    def remove(self, sample: int) -> None:
        with self._lock:
            if sample in self._deleted:
                return  # 重复删除无副作用
            if sample in self._seed_ids:
                raise InvalidState(f"seed sample {sample} cannot be removed")
            if sample not in self._features:
                raise SampleNotFound(f"unknown sample {sample}")
            self._deleted.add(sample)
            self._labels.pop(sample, None)
            self._received.pop(sample, None)
            self._delete_version += 1
            if self._phase == PHASE_LABELING:
                self._check_closure_locked()
            # 训练中的计算会在发布前发现 _delete_version 变化并作废重算。

    def snapshot(self) -> Snapshot:
        with self._lock:
            return Snapshot(
                project_id=self.project_id,
                round=self._round,
                phase=self._phase,
                batch=tuple(self._batch),
                received=dict(self._received),
                models=tuple(info for info, _ in self._models),
                deleted=tuple(sorted(self._deleted)),
                batch_version=self._batch_version,
            )

    # ---------- 内部：状态推进（持锁短临界区） ----------

    def _candidates_locked(self) -> List[int]:
        return [
            pid
            for pid in self._features
            if pid not in self._labels and pid not in self._deleted
        ]

    def _select(self, model: LogisticRegression, cand_ids: List[int]) -> List[int]:
        cand_x = np.asarray([self._features[pid] for pid in cand_ids], dtype=float)
        ranked = _rank_by_entropy(model, cand_ids, cand_x)
        return ranked[: self.k]

    def _check_closure_locked(self) -> None:
        """当前批次所有未删除成员都有标注时闭合批次并触发训练。"""
        if self._phase != PHASE_LABELING:
            return
        if all(m in self._received or m in self._deleted for m in self._batch):
            self._phase = PHASE_TRAINING
            self._schedule_compute_locked()

    def _schedule_compute_locked(self) -> None:
        if self._computing:
            return  # 已有计算在跑；它发布前会基于最新 _delete_version 判断
        self._computing = True
        train_ids = list(self._seed_ids) + sorted(self._labels)
        train_x = np.concatenate(
            [self._seed_x]
            + [self._features[pid].reshape(1, -1) for pid in sorted(self._labels)]
        ) if self._labels else self._seed_x
        train_y = np.concatenate(
            [self._seed_y, np.asarray([self._labels[pid] for pid in sorted(self._labels)])]
        )
        cand_ids = self._candidates_locked()
        cand_x = (
            np.asarray([self._features[pid] for pid in cand_ids], dtype=float)
            if cand_ids
            else np.empty((0, self._seed_x.shape[1]))
        )
        inp = _ComputeInput(
            delete_version=self._delete_version,
            train_ids=train_ids,
            train_x=train_x,
            train_y=train_y,
            cand_ids=cand_ids,
            cand_x=cand_x,
        )
        threading.Thread(
            target=self._compute_and_publish, args=(inp,), daemon=True
        ).start()

    def _publish_locked(
        self, model: LogisticRegression, train_ids: List[int], next_batch: List[int]
    ) -> None:
        """模型版本与下一批一起发布，轮次只增加一次。"""
        version = len(self._models) + 1
        self._models.append((ModelInfo(version, tuple(train_ids)), model))
        self._round += 1
        self._batch = list(next_batch)
        self._received = {}
        self._batch_version = version
        self._phase = PHASE_LABELING if next_batch else PHASE_EXHAUSTED

    # ---------- 内部：后台计算（不持锁） ----------

    def _compute_and_publish(self, inp: _ComputeInput) -> None:
        model = _train_model(inp.train_x, inp.train_y)
        if self._hooks.after_train:
            self._hooks.after_train(self.project_id, list(inp.train_ids))
        ranked = _rank_by_entropy(model, inp.cand_ids, inp.cand_x)
        selected = ranked[: self.k]
        if self._hooks.after_score:
            self._hooks.after_score(self.project_id, list(selected))
        with self._lock:
            self._computing = False
            if inp.delete_version != self._delete_version:
                # 计算期间发生删除：旧结果作废，用最新状态重算。
                self._schedule_compute_locked()
                return
            self._publish_locked(model, inp.train_ids, selected)
