"""SWIFT 按文件加载的 reward 注册入口；普通配置检查不导入此模块。"""
from recipe_sandbox.dandy.training.reward import score_answer, _as_sequence
from typing import Any, List
try:
    from swift.rewards import ORM, orms

    class DataRecipeAccuracy(ORM):
        def __call__(
            self,
            completions: List[Any],
            solution: Any,
            verifier_type: Any,
            choice_options: Any = None,
            **kwargs: Any,
        ) -> List[float]:
            """为一批GRPO completion计算DataRecipe准确率奖励。

            逻辑：将solution和verifier_type对齐到completion批量长度，逐项调用
            score_answer，并保持输入顺序返回奖励。

            参数：
                completions：ms-swift生成的一批候选回答。
                solution：单个标准答案或与回答等长的答案序列。
                verifier_type：单个验证类型或与回答等长的类型序列。
                choice_options：题目中的选项字母到文本映射或该映射的JSON字符串。
                **kwargs：ms-swift额外传入但本奖励不使用的样本字段。

            返回：
                与completions等长的0.0/0.5/1.0奖励列表。
            """
            solutions = _as_sequence(solution, len(completions), "solution")
            verifier_types = _as_sequence(
                verifier_type, len(completions), "verifier_type"
            )
            option_maps = _as_sequence(
                choice_options, len(completions), "choice_options"
            )
            return [
                score_answer(completion, expected, verifier, options)
                for completion, expected, verifier, options in zip(
                    completions, solutions, verifier_types, option_maps
                )
            ]


    orms["datarecipe_accuracy"] = DataRecipeAccuracy
except ImportError:
    # 允许本地在没有ms-swift/NPU环境时导入纯解析函数做静态检查。
    DataRecipeAccuracy = None
