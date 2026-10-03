"""labelwise 的错误类型。"""


class LabelWiseError(Exception):
    """所有 labelwise 错误的基类。"""


class UnknownProjectError(LabelWiseError):
    """项目不存在。"""


class UnknownSampleError(LabelWiseError):
    """样本不属于该项目。"""


class StaleRoundError(LabelWiseError):
    """请求的轮次不是当前轮次，且不是已接受标注的幂等重放。"""


class BatchClosedError(LabelWiseError):
    """当前轮次的批次已闭合（训练中或已耗尽），不再接受新标注。"""


class LabelConflictError(LabelWiseError):
    """同一样本同一轮次提交了与已接受标注不同的标签。"""


class SampleDeletedError(LabelWiseError):
    """样本已被软删除，不能重新生效。"""


class SampleNotInBatchError(LabelWiseError):
    """样本不在当前批次中。"""


class SeedRemovalError(LabelWiseError):
    """种子集样本不可删除。"""
