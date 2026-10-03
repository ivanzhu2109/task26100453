"""LabelWise 并发轮次推进库的自动化测试。

所有交错都通过 Hooks 里的 threading.Event / Barrier 等同步点复现，
不使用 sleep 碰运气；Event.wait 的超时只是失败保护。
"""
import random
import threading

import pytest

from labelwise import Hooks, LabelWise
from labelwise.errors import (
    BatchClosedError,
    LabelConflictError,
    SampleDeletedError,
    SampleNotInBatchError,
    SeedRemovalError,
    StaleRoundError,
    UnknownSampleError,
)

TIMEOUT = 30  # 失败保护，正常路径远快于此


def make_data(n_seed=6, n_pool=40, dim=4, seed=0):
    """构造可分性良好的数据：种子两类，池样本特征定长。"""
    rng = random.Random(seed)
    seed_samples = []
    for i in range(n_seed):
        label = i % 2
        feats = [rng.gauss(3.0 if label else -3.0, 1.0) for _ in range(dim)]
        seed_samples.append((1_000 + i, feats, label))
    pool = []
    for j in range(n_pool):
        base = 3.0 if j % 2 else -3.0
        feats = [rng.gauss(base, 2.0) for _ in range(dim)]
        pool.append((2_000 + j, feats))
    return seed_samples, pool


class Recorder(Hooks):
    """记录所有同步点，并提供基于条件的等待（非 sleep）。"""

    def __init__(self):
        self._cond = threading.Condition()
        self.trains = []   # (pid, round, train_ids)
        self.scores = []   # (pid, round, train_ids, ranked)
        self.pubs = []     # (pid, round, model_version, batch_ids)

    def train_start(self, pid, rnd, train_ids):
        with self._cond:
            self.trains.append((pid, rnd, tuple(train_ids)))
            self._cond.notify_all()

    def score_done(self, pid, rnd, train_ids, ranked):
        with self._cond:
            self.scores.append((pid, rnd, tuple(train_ids), tuple(ranked)))
            self._cond.notify_all()

    def published(self, pid, rnd, version, batch_ids):
        with self._cond:
            self.pubs.append((pid, rnd, version, tuple(batch_ids)))
            self._cond.notify_all()

    def wait_pub(self, pid, rnd):
        with self._cond:
            ok = self._cond.wait_for(
                lambda: any(p[0] == pid and p[1] == rnd for p in self.pubs), TIMEOUT
            )
        assert ok, f"等待项目 {pid} 轮次 {rnd} 发布超时"

    def trains_for(self, pid, rnd):
        return [t for t in self.trains if t[0] == pid and t[1] == rnd]


def submit_batch(lw, pid, rnd, batch):
    for s in batch:
        lw.submit(pid, rnd, s, s % 2)


# ---------------------------------------------------------------- 基本语义
def test_create_validates_and_publishes_first_batch():
    lw = LabelWise()
    seed, pool = make_data(n_pool=25)
    with pytest.raises(ValueError):
        lw.create(seed, pool, 9)
    with pytest.raises(ValueError):
        lw.create(seed, pool, 201)
    with pytest.raises(ValueError):
        lw.create([(1, [0.0], 0), (2, [1.0], 0)], pool, 10)  # 单类别种子
    pid = lw.create(seed, pool, 10)
    snap = lw.snapshot(pid)
    assert snap["round"] == 1 and snap["phase"] == "labeling"
    assert len(snap["batch"]) == 10
    assert len(snap["models"]) == 1
    assert snap["models"][0]["version"] == 1
    assert sorted(snap["models"][0]["train_ids"]) == sorted(s for s, _, _ in seed)
    assert snap["batch_model_version"] == 1


def test_submit_retry_conflict_stale_and_deleted():
    rec = Recorder()
    lw = LabelWise(hooks=rec)
    seed, pool = make_data(n_pool=25)
    pid = lw.create(seed, pool, 10)
    batch = lw.snapshot(pid)["batch"]
    outsider = next(s for s, _ in pool if s not in batch)

    lw.submit(pid, 1, batch[0], 1)
    lw.submit(pid, 1, batch[0], 1)  # 幂等重试
    with pytest.raises(LabelConflictError):
        lw.submit(pid, 1, batch[0], 0)
    with pytest.raises(SampleNotInBatchError):
        lw.submit(pid, 1, outsider, 0)
    with pytest.raises(StaleRoundError):
        lw.submit(pid, 7, batch[1], 0)

    lw.remove(pid, batch[1])
    lw.remove(pid, batch[1])  # 重复删除无副作用
    with pytest.raises(SampleDeletedError):
        lw.submit(pid, 1, batch[1], 0)
    with pytest.raises(SeedRemovalError):
        lw.remove(pid, seed[0][0])
    with pytest.raises(UnknownSampleError):
        lw.remove(pid, 999_999)

    submit_batch(lw, pid, 1, batch[2:])
    rec.wait_pub(pid, 2)
    # 旧轮次：相同标注的幂等重放仍成功，不同标注拒绝，且都不影响当前轮次
    lw.submit(pid, 1, batch[0], 1)
    with pytest.raises(LabelConflictError):
        lw.submit(pid, 1, batch[0], 0)
    with pytest.raises(StaleRoundError):
        lw.submit(pid, 1, outsider, 0)  # 旧轮次且非幂等重放
    assert lw.snapshot(pid)["round"] == 2


def test_exhaustion_publishes_final_model():
    rec = Recorder()
    lw = LabelWise(hooks=rec)
    seed, pool = make_data(n_pool=12)
    pid = lw.create(seed, pool, 10)
    submit_batch(lw, pid, 1, lw.snapshot(pid)["batch"])
    rec.wait_pub(pid, 2)
    snap = lw.snapshot(pid)
    assert len(snap["batch"]) == 2  # 不足 k 条时取剩余全部
    submit_batch(lw, pid, 2, snap["batch"])
    rec.wait_pub(pid, 3)
    snap = lw.snapshot(pid)
    assert snap["phase"] == "exhausted"
    assert snap["batch"] == []
    assert len(snap["models"]) == 3
    assert sorted(snap["models"][2]["train_ids"]) == sorted(
        [s for s, _, _ in seed] + [s for s, _ in pool]
    )
    with pytest.raises(BatchClosedError):
        lw.submit(pid, 3, 2000, 0)


def test_deletion_closes_batch_and_shrinks_train_set():
    rec = Recorder()
    lw = LabelWise(hooks=rec)
    seed, pool = make_data(n_pool=15)
    pid = lw.create(seed, pool, 10)
    batch = lw.snapshot(pid)["batch"]

    submit_batch(lw, pid, 1, batch[:5])
    for s in batch[5:9]:  # 删除 4 个未标注成员，还剩 batch[9] 未标注
        lw.remove(pid, s)
    assert lw.snapshot(pid)["phase"] == "labeling"
    lw.remove(pid, batch[0])  # 删除已标注成员：训练集随之缩小
    assert lw.snapshot(pid)["phase"] == "labeling"
    lw.submit(pid, 1, batch[9], 1)  # 最后一条：自动进入训练
    rec.wait_pub(pid, 2)

    snap = lw.snapshot(pid)
    expected_train = sorted([s for s, _, _ in seed] + batch[1:5] + [batch[9]])
    assert sorted(snap["models"][1]["train_ids"]) == expected_train
    assert snap["phase"] == "labeling"
    assert len(snap["batch"]) == 5  # 15 - 10

    # 删除使批次变空：同样自动推进，无候选则进入耗尽
    for s in snap["batch"]:
        lw.remove(pid, s)
    rec.wait_pub(pid, 3)
    snap = lw.snapshot(pid)
    assert snap["phase"] == "exhausted"
    assert sorted(snap["models"][2]["train_ids"]) == expected_train


# ------------------------------------------------- 必需场景 1：并发最后两条标注
def test_concurrent_final_labels_advance_exactly_once():
    rec = Recorder()
    lw = LabelWise(hooks=rec)
    seed, pool = make_data(n_pool=25)
    pid = lw.create(seed, pool, 10)
    batch = lw.snapshot(pid)["batch"]
    submit_batch(lw, pid, 1, batch[:-2])

    barrier = threading.Barrier(2)
    errors = []

    def worker(sample):
        try:
            barrier.wait(TIMEOUT)
            for _ in range(3):  # 提交 + 相同标注重试
                lw.submit(pid, 1, sample, sample % 2)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(s,)) for s in batch[-2:]]
    for t in threads:
        t.start()
    for t in threads:
        t.join(TIMEOUT)
    assert not errors

    rec.wait_pub(pid, 2)
    snap = lw.snapshot(pid)
    # 只训练一次、只推进一次
    assert len(rec.trains_for(pid, 1)) == 1
    assert len([p for p in rec.pubs if p[0] == pid and p[1] == 2]) == 1
    assert snap["round"] == 2 and snap["phase"] == "labeling"
    assert len(snap["models"]) == 2
    # 训练集没有漏标或重复
    train_ids = snap["models"][1]["train_ids"]
    assert len(train_ids) == len(set(train_ids))
    assert sorted(train_ids) == sorted([s for s, _, _ in seed] + batch)
    # 新批次不含已标注样本
    assert not set(snap["batch"]) & set(batch)


# ------------------------------ 必需场景 2：发布前删除使旧结果作废并重算
def test_delete_before_publish_discards_stale_computation():
    entered = threading.Event()   # 计算完成、尚未发布
    release = threading.Event()   # 放行发布
    state = {}

    class GateHooks(Recorder):
        def score_done(self, pid, rnd, train_ids, ranked):
            super().score_done(pid, rnd, train_ids, ranked)
            if rnd == 1 and not release.is_set():
                state["ranked"] = ranked
                state["train_ids"] = train_ids
                entered.set()
                assert release.wait(TIMEOUT)

    rec = GateHooks()
    lw = LabelWise(hooks=rec)
    seed, pool = make_data(n_pool=30)
    pid = lw.create(seed, pool, 10)
    batch = lw.snapshot(pid)["batch"]
    submit_batch(lw, pid, 1, batch)

    assert entered.wait(TIMEOUT)  # 训练与打分已完成、尚未发布
    victim_train = batch[3]               # 删除一个训练样本（已标注）
    victim_candidate = state["ranked"][0]  # 删除一个高分候选
    lw.remove(pid, victim_train)          # 不持锁计算，删除立即返回
    lw.remove(pid, victim_candidate)
    release.set()

    rec.wait_pub(pid, 2)
    snap = lw.snapshot(pid)
    # 旧结果被作废：重算确实发生，且只发布一次、轮次只加一
    assert len(rec.trains_for(pid, 1)) == 2
    assert len([p for p in rec.pubs if p[0] == pid and p[1] == 2]) == 1
    assert snap["round"] == 2
    assert len(snap["models"]) == 2
    # 重算后的训练集与下一批都排除删除项
    train_ids = snap["models"][1]["train_ids"]
    assert victim_train not in train_ids
    assert sorted(train_ids) == sorted(
        [s for s, _, _ in seed] + [s for s in batch if s != victim_train]
    )
    assert victim_candidate not in snap["batch"]
    assert victim_train not in snap["batch"]
    assert len(snap["batch"]) == 10


# ------------------------------ 必需场景 3：项目间隔离 + 发布原子性
def test_paused_project_does_not_block_others_or_reads():
    a_id = {}
    train_entered = threading.Event()
    train_release = threading.Event()

    class GateHooks(Recorder):
        def train_start(self, pid, rnd, train_ids):
            super().train_start(pid, rnd, train_ids)
            if pid == a_id.get("pid") and rnd == 1 and not train_release.is_set():
                train_entered.set()
                assert train_release.wait(TIMEOUT)

    rec = GateHooks()
    lw = LabelWise(hooks=rec)
    seed_a, pool_a = make_data(n_pool=25, seed=1)
    seed_b, pool_b = make_data(n_pool=25, seed=2)
    a_id["pid"] = lw.create(seed_a, pool_a, 10)
    pid_b = lw.create(seed_b, pool_b, 10)

    submit_batch(lw, a_id["pid"], 1, lw.snapshot(a_id["pid"])["batch"])
    assert train_entered.wait(TIMEOUT)  # 项目 A 的训练被暂停

    # A 暂停期间：删除与读取仍能返回
    pool_only = next(s for s, _ in pool_a if s not in lw.snapshot(a_id["pid"])["batch"])
    lw.remove(a_id["pid"], pool_only)
    snap_a = lw.snapshot(a_id["pid"])
    assert snap_a["phase"] == "training"
    assert pool_only in snap_a["deleted"]

    # 另一个项目仍能完成整轮
    submit_batch(lw, pid_b, 1, lw.snapshot(pid_b)["batch"])
    rec.wait_pub(pid_b, 2)
    snap_b = lw.snapshot(pid_b)
    assert snap_b["round"] == 2 and snap_b["phase"] == "labeling"
    assert len(snap_b["models"]) == 2

    # 恢复 A：整轮完成
    train_release.set()
    rec.wait_pub(a_id["pid"], 2)
    snap_a = lw.snapshot(a_id["pid"])
    assert snap_a["round"] == 2 and snap_a["phase"] == "labeling"
    assert len(snap_a["models"]) == 2

    # 模型版本与批次始终属于同一次发布：每次发布 round == version，
    # 且当前批次由最新模型产出
    for pid in (a_id["pid"], pid_b):
        pubs = [p for p in rec.pubs if p[0] == pid]
        assert [p[1] for p in pubs] == [p[2] for p in pubs]  # round == version
        snap = lw.snapshot(pid)
        assert snap["round"] == len(snap["models"])
        assert snap["batch_model_version"] == snap["models"][-1]["version"]
