"""LabelWise 并发轮次推进测试。

交错复现全部通过 threading.Event / Barrier 同步点完成，不依赖 sleep 碰运气；
等待异步推进使用 wait_until 轮询条件（带超时），仅用于等待结果而非制造交错。
"""
import threading
import time

import numpy as np
import pytest

from labelwise import (
    Hooks,
    InvalidState,
    LabelConflict,
    LabelWise,
    SampleNotFound,
    StaleRound,
)

DIM = 4


def make_seed(n_per_class=10, dim=DIM, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    seed = []
    sid = 1
    for label in (0, 1):
        center = np.full(dim, -1.0 if label == 0 else 1.0)
        for _ in range(n_per_class):
            seed.append((sid, center + rng.normal(0, 0.3, dim), label))
            sid += 1
    return seed


def make_pool(n=30, dim=DIM, rng_seed=1, start_id=1000):
    rng = np.random.default_rng(rng_seed)
    pool = []
    for i in range(n):
        center = np.full(dim, -1.0 if i % 2 == 0 else 1.0)
        pool.append((start_id + i, center + rng.normal(0, 0.5, dim)))
    return pool


def wait_until(cond, timeout=10.0, interval=0.001):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(interval)
    return False


def run_in_thread(fn):
    t = threading.Thread(target=fn, daemon=True)
    t.start()
    return t


# ---------------- 基本行为 ----------------

def test_create_validation():
    lw = LabelWise()
    seed, pool = make_seed(), make_pool()
    with pytest.raises(ValueError):
        lw.create(seed, pool, 9)
    with pytest.raises(ValueError):
        lw.create(seed, pool, 201)
    with pytest.raises(ValueError):  # 单类别种子
        lw.create([(1, [0.0, 0.0], 0), (2, [1.0, 1.0], 0)], pool, 10)
    with pytest.raises(ValueError):  # ID 重复
        lw.create(seed, [(seed[0][0], [0.0] * DIM)] + make_pool(), 10)


def test_create_publishes_first_batch():
    lw = LabelWise()
    pid = lw.create(make_seed(), make_pool(30), 10)
    snap = lw.snapshot(pid)
    assert snap.round == 1
    assert snap.phase == "labeling"
    assert len(snap.batch) == 10
    assert len(snap.models) == 1
    assert snap.models[0].version == 1
    assert set(snap.models[0].train_ids) == {s[0] for s in make_seed()}
    assert snap.batch_version == 1


def test_submit_idempotent_and_conflict_and_stale():
    lw = LabelWise()
    pid = lw.create(make_seed(), make_pool(30), 10)
    snap = lw.snapshot(pid)
    sample = snap.batch[0]
    lw.submit(pid, 1, sample, 1)
    lw.submit(pid, 1, sample, 1)  # 相同标注重试：成功无副作用
    assert lw.snapshot(pid).received == {sample: 1}
    with pytest.raises(LabelConflict):
        lw.submit(pid, 1, sample, 0)
    with pytest.raises(StaleRound):
        lw.submit(pid, 0, snap.batch[1], 0)  # 旧轮次
    with pytest.raises(StaleRound):
        lw.submit(pid, 5, snap.batch[1], 0)  # 未来轮次
    with pytest.raises(SampleNotFound):
        lw.submit(pid, 1, 999999, 0)  # 不在当前批次


def test_remove_semantics():
    lw = LabelWise()
    pid = lw.create(make_seed(), make_pool(30), 10)
    snap = lw.snapshot(pid)
    victim = snap.batch[0]
    lw.remove(pid, victim)
    lw.remove(pid, victim)  # 重复删除无副作用
    assert victim in lw.snapshot(pid).deleted
    with pytest.raises(InvalidState):
        lw.submit(pid, 1, victim, 0)  # 删除后不得重新生效
    with pytest.raises(SampleNotFound):
        lw.remove(pid, 999999)
    seed_id = make_seed()[0][0]
    with pytest.raises(InvalidState):
        lw.remove(pid, seed_id)  # 种子不可删除


def test_exhaustion_publishes_final_model():
    lw = LabelWise()
    pid = lw.create(make_seed(), make_pool(10), 10)  # 池恰好一批
    snap = lw.snapshot(pid)
    for s in snap.batch:
        lw.submit(pid, 1, s, 0)
    assert wait_until(lambda: lw.snapshot(pid).phase == "exhausted")
    snap = lw.snapshot(pid)
    assert snap.round == 2
    assert snap.batch == ()
    assert len(snap.models) == 2  # 最终模型已发布
    with pytest.raises(InvalidState):
        lw.submit(pid, 2, 1000, 0)


def test_delete_completes_batch_closure():
    lw = LabelWise()
    pid = lw.create(make_seed(), make_pool(30), 10)
    batch = lw.snapshot(pid).batch
    for s in batch[:9]:
        lw.submit(pid, 1, s, 1)
    lw.remove(pid, batch[9])  # 删除最后一个未标成员 -> 批次闭合
    assert wait_until(lambda: lw.snapshot(pid).round == 2)
    snap = lw.snapshot(pid)
    assert len(snap.models) == 2
    assert batch[9] not in snap.models[1].train_ids


# ---------------- 交错 1：并发提交最后两条标注 + 重试 ----------------

def test_concurrent_final_submits_single_advance():
    trains = []
    hooks = Hooks(after_train=lambda pid, ids: trains.append(tuple(ids)))
    lw = LabelWise()
    seed, pool = make_seed(), make_pool(30)
    pid = lw.create(seed, pool, 10, hooks=hooks)
    batch = lw.snapshot(pid).batch
    for s in batch[:8]:
        lw.submit(pid, 1, s, 0)

    barrier = threading.Barrier(2)
    errors = []

    def worker(sample, label):
        try:
            barrier.wait(timeout=5)
            lw.submit(pid, 1, sample, label)
            lw.submit(pid, 1, sample, label)  # 重试：幂等
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    t1 = run_in_thread(lambda: worker(batch[8], 0))
    t2 = run_in_thread(lambda: worker(batch[9], 1))
    t1.join(10)
    t2.join(10)
    assert not errors
    assert wait_until(lambda: lw.snapshot(pid).round == 2)

    # 推进后带相同标注的旧轮次重试仍成功且无副作用；不同标注被拒绝
    lw.submit(pid, 1, batch[8], 0)
    with pytest.raises(LabelConflict):
        lw.submit(pid, 1, batch[8], 1)

    snap = lw.snapshot(pid)
    assert snap.round == 2
    assert len(trains) == 1, "只能训练一次"
    assert len(snap.models) == 2, "只能推进/发布一次"
    expected = sorted([s[0] for s in seed] + list(batch))
    assert sorted(snap.models[1].train_ids) == expected  # 训练集无漏标
    assert len(set(snap.models[1].train_ids)) == len(snap.models[1].train_ids)  # 无重复
    assert len(snap.batch) == 10
    assert snap.batch_version == 2


# ---------------- 交错 2：发布前暂停，删除后放行，旧结果作废重算 ----------------

def test_delete_during_paused_publish_discards_stale_result():
    entered_score = threading.Event()
    allow_publish = threading.Event()
    trains, selections = [], []

    def after_train(pid, ids):
        trains.append(tuple(ids))

    def after_score(pid, ids):
        selections.append(tuple(ids))
        entered_score.set()
        allow_publish.wait(timeout=10)

    hooks = Hooks(after_train=after_train, after_score=after_score)
    lw = LabelWise()
    seed, pool = make_seed(), make_pool(30)
    pid = lw.create(seed, pool, 10, hooks=hooks)
    batch = list(lw.snapshot(pid).batch)
    for i, s in enumerate(batch):
        lw.submit(pid, 1, s, i % 2)

    assert entered_score.wait(timeout=10), "计算应暂停在打分完成、尚未发布"
    assert len(selections) == 1
    stale_selection = selections[0]
    victim_train = batch[0]        # 已进入训练集的已标注样本
    victim_cand = stale_selection[0]  # 熵最高的候选

    lw.remove(pid, victim_train)
    lw.remove(pid, victim_cand)
    allow_publish.set()

    assert wait_until(lambda: lw.snapshot(pid).round == 2)
    snap = lw.snapshot(pid)
    assert len(trains) == 2, "受影响的计算应作废重做"
    assert len(snap.models) == 2, "旧结果不得发布，只发布重算结果"
    final_train = snap.models[1].train_ids
    assert victim_train not in final_train
    assert sorted(final_train) == sorted(
        [s[0] for s in seed] + [s for s in batch if s != victim_train]
    )
    assert victim_cand not in snap.batch, "新批次不得包含已删候选"
    assert tuple(snap.batch) != stale_selection, "发布的不得是旧计算结果"
    assert {victim_train, victim_cand} <= set(snap.deleted)


# ---------------- 交错 3：项目互不阻塞，版本与批次同属一次发布 ----------------

def test_projects_are_independent_and_publish_is_atomic():
    entered_train = threading.Event()
    allow_train = threading.Event()
    hooks_a = Hooks(after_train=lambda pid, ids: (entered_train.set(), allow_train.wait(timeout=10)))
    lw = LabelWise()
    pid_a = lw.create(make_seed(rng_seed=0), make_pool(30, rng_seed=1), 10, hooks=hooks_a)
    pid_b = lw.create(make_seed(rng_seed=2), make_pool(30, rng_seed=3), 10)

    batch_a = lw.snapshot(pid_a).batch
    for i, s in enumerate(batch_a):
        lw.submit(pid_a, 1, s, i % 2)
    assert entered_train.wait(timeout=10), "A 的训练应被暂停"
    assert lw.snapshot(pid_a).phase == "training"

    # A 训练暂停期间，A 的删除与快照仍能返回（在子线程执行以验证不阻塞）
    extra = [p for p, _ in make_pool(30, rng_seed=1) if p not in batch_a][0]
    t = run_in_thread(lambda: lw.remove(pid_a, extra))
    t.join(5)
    assert not t.is_alive(), "remove 不得等待训练结束"
    t = run_in_thread(lambda: lw.snapshot(pid_a))
    t.join(5)
    assert not t.is_alive(), "snapshot 不得等待训练结束"

    # B 独立完成整轮
    batch_b = lw.snapshot(pid_b).batch
    for i, s in enumerate(batch_b):
        lw.submit(pid_b, 1, s, i % 2)
    assert wait_until(lambda: lw.snapshot(pid_b).round == 2)
    snap_b = lw.snapshot(pid_b)
    assert len(snap_b.models) == 2
    assert len(snap_b.batch) == 10

    # 恢复 A：删除触发作废重算后最终推进
    allow_train.set()
    assert wait_until(lambda: lw.snapshot(pid_a).round == 2)
    snap_a = lw.snapshot(pid_a)
    assert len(snap_a.models) == 2, "只发布一次"
    assert snap_a.batch_version == snap_a.models[-1].version, "批次与模型版本同属一次发布"
    assert extra not in snap_a.models[-1].train_ids
    assert extra not in snap_a.batch
