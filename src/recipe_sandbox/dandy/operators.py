"""构造 dandy 的原算子子集；只增加比例截断，不复制已有选择算法。"""
import math

from recipe_sandbox.operators import OperatorRegistry, TruncateSamplesOperator
from recipe_sandbox.pipeline.recipe_executor import build_default_operator_registry


class FractionTruncate(TruncateSamplesOperator):
    """保留旧 total_samples 模式；显式 retain_fraction 时按当前输入数量换算。"""

    def transform(self, dataset):
        """校验比例后调用原截断实现，保持原抽样、来源混合及 trace。"""
        if "retain_fraction" in self.config:
            fraction = float(self.config["retain_fraction"])
            if not math.isfinite(fraction) or not 0 < fraction <= 1 or "total_samples" in self.config:
                raise ValueError("retain_fraction 需在 (0,1]，且不能同时指定 total_samples")
            self.config = {**self.config, "total_samples": max(1, int(len(dataset) * fraction))}
        return super().transform(dataset)


def build_registry(names):
    """仅注册启用算子，使执行、warmup、LLM 与 GP 读取同一个算子空间。"""
    original = build_default_operator_registry()
    result = OperatorRegistry()
    for name in names:
        result.register(FractionTruncate if name == "truncate_samples" else original.get(name))
    return result
