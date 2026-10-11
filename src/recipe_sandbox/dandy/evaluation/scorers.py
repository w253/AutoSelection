"""迁自 DataRecipe 的纯答案判分；不加载推理引擎。"""
from typing import Any, Dict
from ..data.answer import classify_answer, normalize_choice_answer, extract_yes_no
from ..training.reward import (completion_text, extract_choice, verify_math,
    has_valid_answer_format, extract_required_answer_segment, score_choice, parse_choice_options)


def answer_types(prediction, record, params):
    """输出 DataRecipe 原判分值和口径；支持半分，不能误记为纯二值正确率。"""
    kind = _score_type(record)
    score, correct, valid, fallback = _score(
        prediction, record["answer"], kind,
        params["invalid_format_fallback_weight"],
        record.get("choice_options") or parse_choice_options(record.get("text_input", "")),
    )
    return {"score": score, "correct": correct, "score_type": kind,
            "format_valid": valid, "fallback_correct": fallback}
def _response_text(response: Any) -> str:
    """从ms-swift对象或字典响应中提取第一条生成文本。

    逻辑：先读取对象式choices[0].message.content，再兼容字典式message.content
    和choice.text；所有已知结构都不匹配时报告错误。

    参数：
        response：VllmEngine返回的单个响应对象或字典。

    返回：
        第一条候选回答文本。
    """
    try:
        return str(response.choices[0].message.content)
    except (AttributeError, IndexError, TypeError):
        pass
    if isinstance(response, dict):
        choices = response.get("choices") or []
        if choices:
            choice = choices[0]
            if isinstance(choice, dict):
                message = choice.get("message") or {}
                if isinstance(message, dict) and message.get("content") is not None:
                    return str(message["content"])
                if choice.get("text") is not None:
                    return str(choice["text"])
    raise ValueError("无法从ms-swift推理结果提取文本")


def _score_type(record: Dict[str, Any]) -> str:
    """根据标准答案确定一个benchmark样本的打分类型。

    逻辑：调用共享分类规则，依次识别单个A至Z字母、大小写不敏感的Yes/No、
    数字或数学公式；其他答案拒绝猜测并报告样本ID。

    参数：
        record：至少包含answer和样本ID的benchmark样本字典。

    返回：
        choice、yes_no或math打分类型字符串。
    """
    answer = record.get("answer", "")
    score_type = classify_answer(answer)
    if score_type is not None:
        return score_type
    raise ValueError(
        "benchmark样本{}的answer不是单个选项字母、Yes/No、数字或数学公式: {!r}".format(
            record.get("id"), answer
        )
    )


def _semantic_correct(prediction: str, answer: Any, score_type: str) -> bool:
    """忽略严格格式门槛，按照答案类型判断预测语义是否正确。

    逻辑：choice先归一化标准答案为裸选项字母、再用extract_choice抽取预测字母后
    比对（MathVision-Wild的答案写作"(A)"，不归一化会永远判错），yes_no调用
    extract_yes_no，math调用verify_math。三个底层提取器保留旧评测的兼容行为：
    有`#####`时优先读取最后一个标记后的片段，没有时从完整回答抽取；未注册类型
    直接报错。

    参数：
        prediction：模型生成文本。
        answer：样本标准答案。
        score_type：由_score_type确定的打分类型。

    返回：
        语义答案正确返回True，否则返回False。
    """
    expected = str(answer).strip()
    if score_type == "choice":
        return extract_choice(prediction) == normalize_choice_answer(expected)
    if score_type == "yes_no":
        return extract_yes_no(prediction) == expected.casefold()
    if score_type == "math":
        return verify_math(prediction, expected)
    raise ValueError("未知打分类型: {}".format(score_type))


def _score(
    prediction: str,
    answer: Any,
    score_type: str,
    invalid_format_fallback_weight: float = 0.0,
    choice_options: Dict[str, str] | None = None,
) -> tuple[float, bool, bool, bool | None]:
    """结合严格格式契约与可选兼容回退计算单条预测得分。

    逻辑：先用extract_required_answer_segment要求全文恰好包含一个`#####`，并要求
    最后一个非空行从行首写成`##### {answer}`。格式合法时只判断该答案段，正确
    选择题字母正确得1分、唯一匹配正确选项文本得0.5分，其余得0分。格式不合法且回退权重大于0时，对原始完整回答调用旧兼容
    语义判断；判断正确仅获得配置权重，判断错误仍为0。权重为0时不运行回退判断，
    从而默认严格拒绝全部格式错误输出。

    参数：
        prediction：模型生成的完整文本。
        answer：样本标准答案。
        score_type：由_score_type确定的choice、yes_no或math类型。
        invalid_format_fallback_weight：格式错误但兼容判断正确时给予的[0, 1]权重。

    返回：
        四元组：实际得分、是否获得满分、严格格式是否合法、回退语义判断结果。
        格式合法或未启用回退时第四项为None；correct仅在严格格式合法且得1分时为True。
    """
    answer_segment = extract_required_answer_segment(prediction)
    if answer_segment is not None:
        score = (
            score_choice(answer_segment, str(answer), choice_options)
            if score_type == "choice"
            else float(_semantic_correct(answer_segment, answer, score_type))
        )
        return score, score == 1.0, True, None
    if invalid_format_fallback_weight == 0.0:
        return 0.0, False, False, None
    fallback_correct = _semantic_correct(prediction, answer, score_type)
    fallback_score = invalid_format_fallback_weight if fallback_correct else 0.0
    return fallback_score, False, False, fallback_correct
