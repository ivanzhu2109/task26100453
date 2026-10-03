"""异常类型定义。"""


class LabelWiseError(Exception):
    """库内所有错误的基类。"""


class ProjectNotFound(LabelWiseError):
    """项目不存在。"""


class SampleNotFound(LabelWiseError):
    """样本不存在于项目中。"""


class LabelConflict(LabelWiseError):
    """同一样本同一轮次提交了不同标注。"""


class StaleRound(LabelWiseError):
    """轮次与当前轮次不符（旧轮次或未来轮次）。"""


class InvalidState(LabelWiseError):
    """当前阶段不允许该操作（如已耗尽、样本已删除）。"""
