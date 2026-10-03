"""计算同步点钩子。

所有钩子都在不持有项目锁的计算线程中调用，因此钩子内部可以安全地
阻塞（测试用它复现交错），不会卡住 submit/remove/snapshot。
"""


class Hooks:
    """默认空实现。测试可子类化或替换以插入可控同步点。"""

    def train_start(self, project_id, round_, train_ids):
        """一次真实训练即将开始（不持锁）。"""

    def score_done(self, project_id, round_, train_ids, ranked_candidate_ids):
        """训练与整池打分已完成、尚未发布（不持锁）。

        ranked_candidate_ids 按预测熵降序、同分按 ID 升序排列。
        """

    def published(self, project_id, round_, model_version, batch_sample_ids):
        """模型版本与下一批已在同一次发布中生效（持锁，禁止阻塞）。"""
