"""LabelWise 并发轮次推进引擎。

设计要点：
- 项目状态（轮次、批次、标注、删除集、模型列表）只在项目锁下读写。
- 模型训练与整池打分在独立计算线程中执行，不持有项目锁；
  submit/remove/snapshot 只短暂持锁，从不等待计算。
- 计算开始时对输入做一致性快照（state_version）；发布前在锁内校验
  快照仍然有效。任何删除都会使 state_version 失效，旧计算结果被作废，
  在锁内重新取样后立即重算，因此删除返回之后的发布绝不含已删除样本。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from math import log

from sklearn.linear_model import LogisticRegression

from .errors import (
    BatchClosedError,
    LabelConflictError,
    SampleDeletedError,
    SampleNotInBatchError,
    SeedRemovalError,
    StaleRoundError,
    UnknownProjectError,
    UnknownSampleError,
)
from .hooks import Hooks

PHASE_LABELING = "labeling"
PHASE_TRAINING = "training"
PHASE_EXHAUSTED = "exhausted"

#: 固定参数的 LogisticRegression
LOGREG_PARAMS = {"max_iter": 1000}

MIN_BATCH = 10
MAX_BATCH = 200


@dataclass(frozen=True)
class ModelRecord:
    version: int
    train_ids: tuple


@dataclass(frozen=True)
class Batch:
    round: int
    model_version: int
    sample_ids: tuple


@dataclass(frozen=True)
class _ComputeInput:
    """一次训练+打分的一致性输入快照。"""

    round: int
    state_version: int
    train_ids: tuple
    train_labels: tuple
    candidate_ids: tuple


@dataclass
class _Project:
    seed_features: dict
    seed_labels: dict
    features: dict
    k: int
    hooks: Hooks
    lock: threading.Lock = field(default_factory=threading.Lock)
    round: int = 0  # create 的种子发布会把它推进到 1
    phase: str = PHASE_LABELING
    batch: Batch | None = None
    labels: dict = field(default_factory=dict)  # 当前轮次已接收标注
    accepted: dict = field(default_factory=dict)  # 所有已接受标注 sample -> label
    accepted_round: dict = field(default_factory=dict)  # sample -> 接受时轮次
    deleted: set = field(default_factory=set)
    models: list = field(default_factory=list)
    state_version: int = 0


def _train_and_rank(proj: _Project, cin: _ComputeInput):
    """真实训练 + 整池打分。纯函数式，不触碰项目状态、不持锁。"""
    x_train = [
        list(proj.seed_features[s]) if s in proj.seed_features else list(proj.features[s])
        for s in cin.train_ids
    ]
    y_train = list(cin.train_labels)
    model = LogisticRegression(**LOGREG_PARAMS)
    model.fit(x_train, y_train)
    proba = model.predict_proba([list(proj.features[s]) for s in cin.candidate_ids]) if cin.candidate_ids else []
    entropy = [-sum(p * log(p) for p in row if p > 0.0) for row in proba]
    order = sorted(range(len(cin.candidate_ids)), key=lambda i: (-entropy[i], cin.candidate_ids[i]))
    ranked = tuple(cin.candidate_ids[i] for i in order)
    return ranked


class LabelWise:
    """并发轮次推进库：标注批次闭合、训练选样与软删除。"""

    def __init__(self, hooks: Hooks | None = None):
        self._hooks = hooks or Hooks()
        self._projects: dict[int, _Project] = {}
        self._registry_lock = threading.Lock()
        self._next_id = 0

    # ------------------------------------------------------------------ create
    def create(self, seed, pool, k) -> int:
        """创建项目，完成种子训练并发布第一批后返回项目 ID。

        seed: [(id, features, label), ...]，必须包含至少两个类别；
        pool: [(id, features), ...]；k: 批量大小，10..200。
        """
        if not (MIN_BATCH <= k <= MAX_BATCH):
            raise ValueError(f"k 必须在 {MIN_BATCH}..{MAX_BATCH} 之间，得到 {k}")
        seed_features, seed_labels = {}, {}
        for sid, feats, label in seed:
            if not isinstance(sid, int):
                raise ValueError(f"样本 ID 必须是整数: {sid!r}")
            if sid in seed_features:
                raise ValueError(f"种子集样本 ID 重复: {sid}")
            seed_features[sid] = tuple(float(x) for x in feats)
            seed_labels[sid] = label
        if len(set(seed_labels.values())) < 2:
            raise ValueError("种子集必须包含至少两个类别")
        features = {}
        for sid, feats in pool:
            if not isinstance(sid, int):
                raise ValueError(f"样本 ID 必须是整数: {sid!r}")
            if sid in seed_features or sid in features:
                raise ValueError(f"样本 ID 重复: {sid}")
            features[sid] = tuple(float(x) for x in feats)
        dims = {len(v) for v in seed_features.values()} | {len(v) for v in features.values()}
        if len(dims) > 1:
            raise ValueError("所有样本的特征维度必须一致")

        proj = _Project(
            seed_features=seed_features,
            seed_labels=seed_labels,
            features=features,
            k=k,
            hooks=self._hooks,
        )
        with self._registry_lock:
            self._next_id += 1
            project_id = self._next_id
            self._projects[project_id] = proj

        # 种子训练 + 第一批发布（create 允许同步完成）
        train_ids = tuple(sorted(seed_features))
        train_labels = tuple(seed_labels[s] for s in train_ids)
        cin = _ComputeInput(0, 0, train_ids, train_labels, tuple(sorted(features)))
        proj.hooks.train_start(project_id, 0, train_ids)
        ranked = _train_and_rank(proj, cin)
        proj.hooks.score_done(project_id, 0, train_ids, ranked)
        with proj.lock:
            self._publish_locked(proj, project_id, cin, ranked)
        return project_id

    # ------------------------------------------------------------------ submit
    def submit(self, project, round, sample, label) -> None:
        """提交指定轮次的标注。相同标注重试幂等成功，不同标注拒绝。"""
        proj = self._get(project)
        with proj.lock:
            if proj.accepted_round.get(sample) == round:
                if proj.accepted[sample] == label:
                    return  # 幂等重试：成功且无副作用
                raise LabelConflictError(
                    f"样本 {sample} 在轮次 {round} 已接受标注 {proj.accepted[sample]!r}，拒绝 {label!r}"
                )
            if round != proj.round:
                raise StaleRoundError(f"当前轮次为 {proj.round}，拒绝轮次 {round} 的提交")
            if proj.phase != PHASE_LABELING:
                raise BatchClosedError(f"轮次 {round} 的批次已闭合（阶段 {proj.phase}）")
            if sample in proj.deleted:
                raise SampleDeletedError(f"样本 {sample} 已删除，不能重新生效")
            if proj.batch is None or sample not in proj.batch.sample_ids:
                raise SampleNotInBatchError(f"样本 {sample} 不在轮次 {round} 的批次中")
            proj.labels[sample] = label
            proj.accepted[sample] = label
            proj.accepted_round[sample] = round
            proj.state_version += 1
            if self._batch_closed_locked(proj):
                self._start_training_locked(proj, project)

    # ------------------------------------------------------------------ remove
    def remove(self, project, sample) -> None:
        """软删除池中的样本（含已标注样本）。返回时删除已生效；重复删除无副作用。"""
        proj = self._get(project)
        with proj.lock:
            if sample in proj.seed_features:
                raise SeedRemovalError(f"种子集样本 {sample} 不可删除")
            if sample not in proj.features:
                raise UnknownSampleError(f"样本 {sample} 不属于项目 {project}")
            if sample in proj.deleted:
                return
            proj.deleted.add(sample)
            proj.state_version += 1
            if proj.phase == PHASE_LABELING and self._batch_closed_locked(proj):
                self._start_training_locked(proj, project)

    # ---------------------------------------------------------------- snapshot
    def snapshot(self, project) -> dict:
        """返回项目的一致视图。"""
        proj = self._get(project)
        with proj.lock:
            return {
                "round": proj.round,
                "phase": proj.phase,
                "batch": list(proj.batch.sample_ids) if proj.batch else [],
                "batch_model_version": proj.batch.model_version if proj.batch else None,
                "labels": dict(proj.labels),
                "deleted": sorted(proj.deleted),
                "models": [
                    {"version": m.version, "train_ids": list(m.train_ids)} for m in proj.models
                ],
            }

    # ------------------------------------------------------------- 内部：状态
    def _get(self, project) -> _Project:
        with self._registry_lock:
            proj = self._projects.get(project)
        if proj is None:
            raise UnknownProjectError(f"项目 {project} 不存在")
        return proj

    @staticmethod
    def _batch_closed_locked(proj: _Project) -> bool:
        """当前批次所有未删除成员都已有标注（空批次视为闭合）。"""
        if proj.batch is None:
            return False
        return all(s in proj.labels or s in proj.deleted for s in proj.batch.sample_ids)

    def _snapshot_input_locked(self, proj: _Project) -> _ComputeInput:
        train_ids = tuple(
            sorted(
                list(proj.seed_features)
                + [s for s in proj.accepted if s not in proj.deleted]
            )
        )
        train_labels = tuple(
            proj.seed_labels[s] if s in proj.seed_labels else proj.accepted[s] for s in train_ids
        )
        candidate_ids = tuple(
            sorted(s for s in proj.features if s not in proj.deleted and s not in proj.accepted)
        )
        return _ComputeInput(proj.round, proj.state_version, train_ids, train_labels, candidate_ids)

    def _start_training_locked(self, proj: _Project, project_id) -> None:
        """批次闭合：转入训练阶段并启动后台计算（锁内调用，立即返回）。"""
        proj.phase = PHASE_TRAINING
        cin = self._snapshot_input_locked(proj)
        thread = threading.Thread(
            target=self._compute_loop,
            args=(proj, project_id, cin),
            name=f"labelwise-compute-{project_id}-r{cin.round}",
            daemon=True,
        )
        thread.start()

    # ------------------------------------------------------------- 内部：计算
    def _compute_loop(self, proj: _Project, project_id, cin: _ComputeInput) -> None:
        """训练+打分（不持锁），发布前校验一致性；失效则重算直至可发布。"""
        while True:
            proj.hooks.train_start(project_id, cin.round, cin.train_ids)
            ranked = _train_and_rank(proj, cin)
            proj.hooks.score_done(project_id, cin.round, cin.train_ids, ranked)
            with proj.lock:
                valid = (
                    proj.phase == PHASE_TRAINING
                    and proj.round == cin.round
                    and proj.state_version == cin.state_version
                )
                if valid:
                    self._publish_locked(proj, project_id, cin, ranked)
                    return
                if proj.phase != PHASE_TRAINING or proj.round != cin.round:
                    return  # 该轮次已被其他路径推进，作废
                cin = self._snapshot_input_locked(proj)  # 删除使快照失效：重取重算

    def _publish_locked(self, proj: _Project, project_id, cin: _ComputeInput, ranked) -> None:
        """模型版本与下一批在同一次锁内发布，轮次只增加一次。"""
        version = len(proj.models) + 1
        proj.models.append(ModelRecord(version, cin.train_ids))
        next_ids = tuple(s for s in ranked if s not in proj.deleted)[: proj.k]
        proj.round += 1
        proj.labels = {}
        if next_ids:
            proj.batch = Batch(proj.round, version, next_ids)
            proj.phase = PHASE_LABELING
        else:
            proj.batch = None
            proj.phase = PHASE_EXHAUSTED
        proj.hooks.published(project_id, proj.round, version, next_ids)
