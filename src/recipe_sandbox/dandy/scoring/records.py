"""图文前向的充分统计与稀疏行，不实现第二套缓存。"""
from dataclasses import dataclass
from typing import Tuple
import math

@dataclass(frozen=True)
class MeanStatistic:
    """保存一个非负逐token量的总和、计数和派生均值。

    ``total``必须为有限非负数，``count``必须为正整数。对象用于保存NLL和
    Varentropy的充分统计量，使候选聚合或消融可以复算均值而不是只依赖最终分数。
    """

    total: float
    count: int

    def __post_init__(self) -> None:
        """校验总和与计数，拒绝NaN、Inf、负数或空统计量。

        参数：
            无；读取当前对象字段。

        返回：
            无；字段非法时抛出ValueError。
        """
        if (
            isinstance(self.total, bool)
            or not isinstance(self.total, (int, float))
            or not math.isfinite(float(self.total))
            or self.total < 0
        ):
            raise ValueError("逐token统计总和必须是有限非负数")
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count <= 0:
            raise ValueError("逐token统计计数必须是正整数")

    @property
    def mean(self) -> float:
        """返回总和除以计数得到的有限均值。

        参数：
            无。

        返回：
            当前充分统计量对应的浮点均值。
        """
        return float(self.total) / self.count

@dataclass(frozen=True)
class SparseFeatureRow:
    """保存一条样本最多192个非零SAE维度和值。

    indices与values保持按激活值选择后的顺序且长度相同。该对象只表达样本特征，
    不记录计算设备或分布式rank。
    """

    sample_id: str
    indices: Tuple[int, ...]
    values: Tuple[float, ...]
