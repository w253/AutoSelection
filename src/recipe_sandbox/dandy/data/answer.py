"""统一定义DataRecipe benchmark答案的可评分类型和保守提取规则。

本模块只依赖Python标准库，供六个benchmark formatter与evaluate.py共同使用。
它集中定义统一回答格式指令及其文本拼接方式，按照单字母选项、Yes/No、数字或
数学公式的顺序分类标准答案，并提供保守的Yes/No预测提取。formatter据此在采样前
排除无法评分的记录，评测代码据此选择确定性规则；模块不读取文件、不修改输入记录，
也不依赖NPU或ms-swift环境。选择题标准答案在比对前统一用
normalize_choice_answer折叠为裸选项字母。

训练与评测统一要求模型把最终答案写在标记`#####`之后。兼容提取函数会优先只
判定最后一个标记之后的片段，标记不存在时退回全文；eval和GRPO reward默认在
调用它们之前另行执行严格格式校验，仅显式启用评测回退权重时才使用兼容行为。
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Iterable, List, Tuple


CHOICE_ANSWER_PATTERN = re.compile(
    r"^[\(\[]?([A-Z])[\)\]]?$",
    flags=re.IGNORECASE,
)
# 选项字母后可跟一个简短值/表达式，如"(B) 6"、"(D) IV"、"(E) e"；仅接受大写字母，
# 避免把数学答案里的单个小写变量（如"x"）误判为选项。
CHOICE_WITH_VALUE_PATTERN = re.compile(r"^[\(\[]?([A-Z])[\)\]]?[\s:.)\-–]+(\S{1,12})$")
# 选项字母前缀：覆盖"(A)"、"(B) 6"、"(A) Vanuatu"以及裸字母"A"。"x2"这类变量不含
# 分隔符，因此不会被匹配。用于把选择题标准答案归一化为裸选项字母。
CHOICE_PREFIX_PATTERN = re.compile(r"^[\(\[]?([A-Za-z])[\)\]]?(?=$|[\s:.)\-–])")
# 带外层括号的选项前缀：选项文本可能很长或含空格，如"(B) I, III and V"、
# "(A) All three strips have the same area"。这类记录本质是选择题，若因选项文本
# 过长或含空格而落到数学分支，就会被交给数学等价比较而永远判错。
BRACKETED_CHOICE_PREFIX_PATTERN = re.compile(r"^[\(\[]([A-Z])[\)\]](?=$|[\s:.)\-–])")
YES_NO_ANSWER_PATTERN = re.compile(r"^(yes|no)$", flags=re.IGNORECASE)
BARE_YES_NO_PATTERN = re.compile(
    r"^\s*(yes|no)[.!]?\s*$",
    flags=re.IGNORECASE,
)
TAGGED_YES_NO_PATTERN = re.compile(
    r"<answer>\s*(yes|no)\s*</answer>",
    flags=re.IGNORECASE | re.DOTALL,
)
EXPLICIT_YES_NO_PATTERN = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is|:)?\s*(yes|no)\b",
    flags=re.IGNORECASE,
)
MATH_SYNTAX_PATTERN = re.compile(r"[=+\-*/^_<>%()[\]{},:]")

# 最终答案标记；训练数据、奖励函数、评测抽取三处必须保持一致。
ANSWER_MARKER = "#####"
OLD_FINAL_ANSWER_EXAMPLE = "For example: ##### 70 or ##### A."
FINAL_ANSWER_EXAMPLE = (
    "For multiple-choice questions, write only the option letter, for example ##### C. "
    "For other questions, write the exact number or expression, for example ##### 70."
)
FINAL_ANSWER_INSTRUCTION = (
    "Response format:\n"
    "Please reason step by step. End your response with exactly one final line in the "
    "form ##### {answer}, replacing {answer} with the answer and omitting the braces. "
    "Do not write anything after that line. " + FINAL_ANSWER_EXAMPLE
)
FINAL_ANSWER_SECTION_SEPARATOR = "\n\n---\n\n"


def strengthen_final_answer_instruction(text: str) -> str:
    """在读取已格式化的旧数据时更新选项字母提示，不改写题目其他内容。"""
    return text.replace(OLD_FINAL_ANSWER_EXAMPLE, FINAL_ANSWER_EXAMPLE)


def append_final_answer_instruction(text: Any) -> str:
    """在benchmark原始输入后追加统一回答格式区块。

    逻辑：把输入转换为字符串并去除首尾空白；非空题干后使用两个空行、Markdown
    分隔线和两个空行追加带`Response format:`标题的共享指令，使该区块不会与题目
    末尾的候选选项混在一起。空输入直接返回同一共享指令，供题目完全位于图片中的
    benchmark使用。

    参数：
        text：原始题干、题干与选项的组合文本，或空值。

    返回：
        包含且仅包含一次统一回答格式指令的text_input字符串。
    """
    normalized = str(text).strip() if text is not None else ""
    if not normalized:
        return FINAL_ANSWER_INSTRUCTION
    return "{}{}{}".format(
        normalized,
        FINAL_ANSWER_SECTION_SEPARATOR,
        FINAL_ANSWER_INSTRUCTION,
    )


def strip_answer_marker(text: str) -> str:
    """把带最终答案标记的回答折叠为标记之后的答案片段。

    逻辑：存在标记时只保留最后一个标记之后的文本并去除首尾空白；标记之后为空
    时保留原文，避免把空串交给下游判定；没有标记时原样返回，兼容旧模型与旧日志。

    参数：
        text：模型生成的完整回答文本。

    返回：
        用于答案抽取的文本片段。
    """
    if ANSWER_MARKER not in text:
        return text
    tail = text.rsplit(ANSWER_MARKER, 1)[1].strip()
    return tail if tail else text


def looks_like_math_answer(answer: str) -> bool:
    """保守判断一个非选项、非Yes/No答案是否表现为数字或数学公式。

    逻辑：包含数字、LaTeX命令、Unicode数学符号或常见公式结构字符时判定为
    数学答案。空文本、普通自然语言以及单个英文字母不在本函数中判为数学题；
    单字母由classify_answer优先按选项处理。

    输入：
        answer：已经转换为字符串的标准答案。

    输出：
        看起来是数字或数学公式时返回True，否则返回False。
    """
    normalized = answer.strip()
    if not normalized:
        return False
    if any(character.isdigit() for character in normalized):
        return True
    if "\\" in normalized:
        return True
    if MATH_SYNTAX_PATTERN.search(normalized):
        return True
    return any(
        unicodedata.category(character) == "Sm"
        for character in normalized
    )


def looks_like_choice_with_value(answer: str) -> bool:
    """判断"选项字母+简短值"形态的标准答案。

    逻辑：匹配"可带括号的大写选项字母 + 分隔符 + 不超过12字符的值"，并要求
    该值不是长度大于2的纯字母单词（那样更可能是自然语言而不是选项值）。
    MathVision-Wild一类数据集的答案写作"(A)"、"(B) 6"、"(D) IV"，这些记录
    本质是选择题，若不识别就会被误送到数学等价比较。

    输入：
        answer：已经去除首尾空白的标准答案字符串。

    输出：
        形如选项(+值)时返回True，否则返回False。
    """
    match = CHOICE_WITH_VALUE_PATTERN.fullmatch(answer)
    if match is None:
        return False
    value = match.group(2)
    if value.isalpha() and len(value) > 2:
        return False
    return True


def classify_answer(answer: Any) -> str | None:
    """把标准答案分类为评测代码能够确定性处理的类型。

    逻辑：将答案转为去除首尾空白的字符串，依次匹配单个A至Z选项（允许括号）、
    "选项字母+简短值"、"括号选项字母+任意选项文本"、大小写不敏感的Yes/No，以及
    数字或数学公式；任何规则都不满足时返回None，不把自然语言答案擅自交给模糊
    匹配。

    输入：
        answer：formatter记录中的标准答案，可为任意可转成字符串的值。

    输出：
        返回choice、yes_no或math；无法可靠评分时返回None。
    """
    normalized = str(answer).strip()
    if CHOICE_ANSWER_PATTERN.fullmatch(normalized):
        return "choice"
    if looks_like_choice_with_value(normalized):
        return "choice"
    if BRACKETED_CHOICE_PREFIX_PATTERN.match(normalized):
        return "choice"
    if YES_NO_ANSWER_PATTERN.fullmatch(normalized):
        return "yes_no"
    if looks_like_math_answer(normalized):
        return "math"
    return None


def normalize_choice_answer(answer: Any) -> str:
    """把选择题标准答案归一化为单个大写选项字母。

    逻辑：去除首尾空白后匹配选项字母前缀，命中则返回该字母的大写形式。MathVision-Wild
    一类数据集的答案写作"(A)"或"(B) 6"，而抽取函数返回的是裸字母；两边不归一化时
    "(A)"永远不等于"A"，这些记录无论模型答对与否都被判错。不匹配任何选项前缀的
    答案原样返回，因此本就写作裸字母的MMMU-Pro、MMStar等数据集行为不变。

    输入：
        answer：formatter记录中的标准答案，可为任意可转成字符串的值。

    输出：
        裸大写选项字母，或未归一化的原答案。
    """
    normalized = str(answer).strip()
    match = CHOICE_PREFIX_PATTERN.match(normalized)
    if match is None:
        return normalized
    return match.group(1).upper()


def partition_supported_answers(
    records: Iterable[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """按统一答案规则把benchmark记录拆分为可评分与不可评分两组。

    逻辑：保持输入迭代顺序，逐条读取answer并调用classify_answer；分类成功的
    记录放入第一组，失败的记录放入第二组。函数保留原字典对象，不复制或修改
    字段，以便formatter继续使用已经生成的稳定ID和内部字段。

    输入：
        records：包含answer字段的benchmark记录迭代器。

    输出：
        二元组，依次为可评分记录列表和被过滤记录列表。
    """
    supported_records: List[Dict[str, Any]] = []
    rejected_records: List[Dict[str, Any]] = []
    for record in records:
        destination = (
            supported_records
            if classify_answer(record.get("answer")) is not None
            else rejected_records
        )
        destination.append(record)
    return supported_records, rejected_records


def extract_yes_no(text: str) -> str | None:
    """从模型回答中保守提取最终Yes或No。

    逻辑：先折叠到最终答案标记之后的片段；再依次接受只含Yes/No的裸回答、
    answer标签中的答案，以及带有明确Final answer或Answer提示词的表达；每种
    结构存在多个匹配时取最后一个，不从无答案标记的任意推理正文中抓取Yes或No。

    输入：
        text：模型生成的完整回答文本。

    输出：
        规范化的小写yes或no；无法可靠提取时返回None。
    """
    text = strip_answer_marker(text)
    bare_match = BARE_YES_NO_PATTERN.fullmatch(text)
    if bare_match:
        return bare_match.group(1).casefold()
    tagged_matches = TAGGED_YES_NO_PATTERN.findall(text)
    if tagged_matches:
        return tagged_matches[-1].casefold()
    explicit_matches = EXPLICIT_YES_NO_PATTERN.findall(text)
    return explicit_matches[-1].casefold() if explicit_matches else None
