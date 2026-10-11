"""实现并注册DataRecipe的ms-swift GRPO确定性最终答案奖励。

本模块统一读取字符串或消息结构的completion，根据样本verifier_type执行选择题
抽取或数学等价验证，并返回0/0.5/1奖励。GRPO奖励要求回答中恰好存在一个最终答案
标记，并要求最后一个非空行写成`##### {answer}`，只对该末行答案评分；格式
合法性作为正确性评分的前置门槛，不注册成独立优化奖励。选择题标准答案先归一化
为裸选项字母，数学题
两侧的裸LaTeX补上定界符后再交给math-verify。模块也向评测代码暴露兼容旧输出的
选择题抽取和数学验证函数；没有ms-swift时仍可安全导入。
"""

from __future__ import annotations

import json
import re
from typing import Any, List, Sequence


CHOICE_PATTERN = re.compile(
    r"(?i)(?:final\s+answer|answer|option|choice)\s*(?:is|:)?\s*\(?([A-Z])\b\)?"
)
TAGGED_CHOICE_PATTERN = re.compile(
    r"(?is)<answer>\s*\(?([A-Z])\b\)?\s*</answer>"
)
BARE_CHOICE_PATTERN = re.compile(r"^\s*\(?([A-Z])\)?[.)]?\s*$", flags=re.IGNORECASE)
# 末行选项字母形态，三种：
#   "(E) The image does not..."：带外层括号的选项；
#   "D: The suitcase is..."：紧跟冒号或右括号的 A–J；
#   "B. 50°"、"C. Vanuatu"：字母加句点再跟内容——SFT数据的CONCLUSION里大量出现。
# 句点形态要求整行以该字母开头（只允许前面有一个左括号），否则人名缩写
# （"Robert G. Hagstrom"）会被误当选项；句末的"A."由BARE_CHOICE_PATTERN负责。
TAIL_LINE_CHOICE_PATTERN = re.compile(
    r"(?:\(([A-J])\)"
    r"|(?<![A-Za-z])([A-J])[):](?=\s|$)"
    r"|^[\(\[]?([A-J])\.(?=\s+\S))"
)
# 选项字母前缀：把"(A)"、"(B) 6"、"(A) Vanuatu"和裸字母统一折叠为单个大写字母。
CHOICE_PREFIX_PATTERN = re.compile(r"^[\(\[]?([A-Za-z])[\)\]]?(?=$|[\s:.)\-–])")
# \boxed{C}、\boxed{(C)}这类把选项包进\boxed的写法，仅作最后兜底。
BOXED_CHOICE_PATTERN = re.compile(r"\\boxed\s*\{\s*\(?([A-Z])\)?\s*\}", flags=re.IGNORECASE)
# 无定界符的裸LaTeX：math-verify只抽取$...$、\(...\)或\boxed{}里的公式。
LATEX_COMMAND_PATTERN = re.compile(r"\\[A-Za-z]+")
# 上下标"^{...}"、"_{...}"同样是LaTeX语法，裸写时也要一起补定界符。
SUPERSCRIPT_PATTERN = re.compile(r"[\^_]\s*\{")
LATEX_DELIMITER_PATTERN = re.compile(r"\$|\\\(|\\\[|\\boxed")

# 最终答案标记；训练数据、answer.py与评测三处必须保持一致。这里保留独立实现而
# 不从answer.py导入，使本模块仍可作为独立插件被ms-swift按文件路径加载。
ANSWER_MARKER = "#####"
CHOICE_LINE_PATTERN = re.compile(r"^\s*(?:\(([A-Z])\)|([A-Z])[.):]?)[ \t]+(.+?)\s*$")


def parse_choice_options(text: str) -> dict[str, str]:
    """从题目中读取独立行上的选项；重复字母或重复内容不用于半分判定。

    只解析题目正文中A至Z开头的选项行，不从模型推理或图片猜测选项。
    返回的字典供GRPO数据额外字段和benchmark判分共用。
    """
    options: dict[str, str] = {}
    for line in text.splitlines():
        if line.strip().casefold() == "response format:":
            break
        match = CHOICE_LINE_PATTERN.match(line)
        if match:
            letter = match.group(1) or match.group(2)
            if letter in options:
                return {}
            options[letter] = match.group(3).strip()
    return options


def _normalized_option_text(text: str) -> str:
    """仅合并空白和大小写，不删除数字、单位或数学符号。"""
    return " ".join(text.casefold().split()).rstrip(".。 ")


def score_choice(answer_segment: str, expected: str, options: Any = None) -> float:
    """比较答案片段与标准选项，返回1、0.5或0。

    options接受原始字典或训练JSONL中的JSON字符串；后者先解码为同一映射，
    不改变字母及唯一选项内容匹配规则。非法JSON抛出解析错误，便于定位输入问题。
    """
    # 训练JSONL使用字符串避免Arrow选项struct随题目变化；评测仍可直接传dict。
    if isinstance(options, str):
        options = json.loads(options)
    expected_letter = normalize_choice_answer(expected)
    extracted = extract_choice(answer_segment)
    if extracted is not None:
        return float(extracted == expected_letter)
    if not isinstance(options, dict) or len(options) < 2 or expected_letter not in options:
        return 0.0
    answer_text = _normalized_option_text(answer_segment)
    matches = [
        letter for letter, content in options.items()
        if answer_text == _normalized_option_text(str(content))
    ]
    return 0.5 if matches == [expected_letter] else 0.0


def strip_answer_marker(text: str) -> str:
    """把带最终答案标记的回答折叠为标记之后的答案片段。

    逻辑与answer.strip_answer_marker一致：存在标记时只保留最后一个标记之后的
    文本并去除首尾空白；标记之后为空时保留原文；没有标记时原样返回。

    参数：
        text：模型生成的完整回答文本。

    返回：
        用于答案抽取的文本片段。
    """
    if ANSWER_MARKER not in text:
        return text
    tail = text.rsplit(ANSWER_MARKER, 1)[1].strip()
    return tail if tail else text


def extract_required_answer_segment(text: str) -> str | None:
    """提取满足GRPO奖励格式契约的末行最终答案。

    逻辑：先去除回答末尾的空白字符，再要求全文恰好出现一次`#####`；最后一个
    非空行必须从行首写成`##### {answer}`，标记后至少有一个空格，答案
    必须非空。满足时仅返回该末行中的答案；缺少或重复标记、标记不在末行行首、
    标记后没有空白、答案为空，或标记后仍有其他正文时都返回None。这样奖励不会
    从推理正文、早先候选答案或被截断的回答中误取正确值。

    参数：
        text：已经转换为普通字符串的完整模型回答。

    返回：
        合法时返回末行标记后的答案；格式不合法时返回None。
    """
    if text.count(ANSWER_MARKER) != 1:
        return None
    stripped_text = text.rstrip()
    if not stripped_text:
        return None
    final_line = stripped_text.splitlines()[-1]
    required_prefix = "{} ".format(ANSWER_MARKER)
    if not final_line.startswith(required_prefix):
        return None
    answer_segment = final_line[len(required_prefix):].strip()
    return answer_segment or None


def has_valid_answer_format(completion: Any) -> bool:
    """判断一个completion是否满足GRPO最终答案格式契约。

    逻辑：先用completion_text兼容字符串、消息字典和消息列表，再调用
    extract_required_answer_segment检查唯一标记及末行`##### {answer}`契约。本函数
    只暴露格式状态供日志或后续实验使用，不自行注册为独立奖励。

    参数：
        completion：字符串、消息字典、消息列表或其他可转为字符串的对象。

    返回：
        最后一个非空行满足`##### {answer}`且全文只有一个标记时返回True，否则
        返回False。
    """
    return extract_required_answer_segment(completion_text(completion)) is not None


def normalize_choice_answer(answer: Any) -> str:
    """把选择题标准答案归一化为单个大写选项字母。

    逻辑与answer.normalize_choice_answer一致：去除首尾空白后匹配选项字母前缀，
    命中则返回该字母的大写形式，否则原样返回。本模块保留独立实现而不从answer.py
    导入，使它能继续作为单文件插件被ms-swift按路径加载；两处规则必须同步修改。

    参数：
        answer：样本的标准最终答案。

    返回：
        裸大写选项字母，或未归一化的原答案。
    """
    normalized = str(answer).strip()
    match = CHOICE_PREFIX_PATTERN.match(normalized)
    if match is None:
        return normalized
    return match.group(1).upper()


def completion_text(completion: Any) -> str:
    """将ms-swift可能返回的completion结构统一转换为文本。

    逻辑：字符串直接返回；消息列表优先只取最后一条assistant消息的content，避免
    把user提示词中的格式示例误计为模型输出；列表没有角色信息时才连接其中全部
    content。单个消息字典读取content，其他对象使用str转换，避免评分入口依赖
    某一种固定返回结构。

    参数：
        completion：字符串、消息字典、消息列表或其他可转为字符串的对象。

    返回：
        用于答案抽取和验证的普通字符串。
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        assistant_contents = [
            str(message["content"])
            for message in completion
            if (
                isinstance(message, dict)
                and str(message.get("role", "")).casefold() == "assistant"
                and message.get("content") is not None
            )
        ]
        if assistant_contents:
            return assistant_contents[-1]
        contents = []
        for message in completion:
            if isinstance(message, dict) and message.get("content") is not None:
                contents.append(str(message["content"]))
        return "\n".join(contents)
    if isinstance(completion, dict) and completion.get("content") is not None:
        return str(completion["content"])
    return str(completion)


def extract_choice(text: str) -> str | None:
    """从模型回答中保守提取最后一个选择题选项字母。

    逻辑：先折叠到最终答案标记之后的片段；再依次匹配纯字母回答、answer标签和
    带Answer/Option等提示词的表达；随后只在最后一个非空行里找末行选项字母形态
    （模型常以"D: …"、"(E) …"或"B. 50°"给出最终选项），最后退回全文的提示词
    匹配。最后两条兜底规则——末行只写一个字母、答案被\\boxed{}包裹——只在前述规则
    全都没有结果时才生效，因此不会改变任何已有的判定结果。其余情况不会从任意正文中
    无条件抓取单个大写字母，以减少推理文本误判。

    参数：
        text：模型生成的完整回答文本。

    返回：
        规范化的大写A至Z；无法可靠提取时返回None。
    """
    text = strip_answer_marker(text)
    bare_match = BARE_CHOICE_PATTERN.fullmatch(text)
    if bare_match:
        return bare_match.group(1).upper()
    tagged_matches = TAGGED_CHOICE_PATTERN.findall(text)
    if tagged_matches:
        return tagged_matches[-1].upper()

    tail_lines = [line for line in text.splitlines() if line.strip()]
    if tail_lines:
        tail_line = tail_lines[-1]
        tail_keyword_matches = CHOICE_PATTERN.findall(tail_line)
        if tail_keyword_matches:
            return tail_keyword_matches[-1].upper()
        tail_matches = TAIL_LINE_CHOICE_PATTERN.findall(tail_line)
        if tail_matches:
            bracketed, punctuated, dotted = tail_matches[0]
            return (bracketed or punctuated or dotted).upper()

    matches = CHOICE_PATTERN.findall(text)
    if matches:
        return matches[-1].upper()

    # 兜底一：末行只写一个裸字母（prompt未要求#####格式时模型可能这样收尾）。
    if tail_lines:
        bare_tail = BARE_CHOICE_PATTERN.fullmatch(tail_lines[-1])
        if bare_tail:
            return bare_tail.group(1).upper()

    # 兜底二：答案被\boxed{}包裹。
    boxed_matches = BOXED_CHOICE_PATTERN.findall(text)
    return boxed_matches[-1].upper() if boxed_matches else None


def wrap_bare_latex(text: str) -> str:
    """给没有数学定界符的裸LaTeX表达式补上一对美元符号。

    逻辑：math-verify的LaTeX抽取器只识别$...$、\\(...\\)或\\boxed{}包裹的公式，而
    训练数据与模型回答常直接写出"\\frac{\\sqrt{2}}{2}"、"4\\pi"这类裸表达式。不包
    裹时金标会解析为空（该样本永远拿不到奖励），或只解析出开头的数字——"4\\pi"被
    当成4——从而把答错的回答判成正确；"2^{115}"这样只有上下标的裸写法会被只解析成
    开头的2。只有文本含LaTeX命令或上下标且完全不含现有定界符时才补包装，已带定界符
    或纯数字文本原样返回。

    参数：
        text：待解析的答案文本。

    返回：
        可直接交给math-verify解析的文本。
    """
    if not (
        LATEX_COMMAND_PATTERN.search(text) or SUPERSCRIPT_PATTERN.search(text)
    ):
        return text
    if LATEX_DELIMITER_PATTERN.search(text):
        return text
    return "${}$".format(text.strip())


def verify_math(prediction: str, solution: str) -> bool:
    """使用math-verify判断预测与标准答案是否数学等价。

    逻辑：先折叠到最终答案标记之后的片段；延迟导入math-verify，给两侧的裸LaTeX补上
    定界符后，分别用LaTeX和普通表达式提取器解析预测与答案，在5秒解析时限下调用
    verify比较。正常的解析失败返回False；依赖未安装或API调用异常会明确抛出，避免
    把评测环境故障静默统计为模型答错。

    参数：
        prediction：模型生成的答案文本。
        solution：数据样本中的标准数学答案。

    返回：
        数学等价返回True，否则返回False。
    """
    prediction = strip_answer_marker(prediction)
    try:
        from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
    except ImportError as error:
        raise RuntimeError(
            "数学评分需要安装math-verify及兼容的ANTLR运行时"
        ) from error

    extraction_config = [LatexExtractionConfig(), ExprExtractionConfig()]
    parsed_prediction = parse(
        wrap_bare_latex(prediction),
        extraction_config=extraction_config,
        parsing_timeout=5,
        raise_on_error=False,
    )
    parsed_solution = parse(
        wrap_bare_latex(solution),
        extraction_config=extraction_config,
        parsing_timeout=5,
        raise_on_error=False,
    )
    if not parsed_prediction or not parsed_solution:
        return False
    return bool(
        verify(
            parsed_solution,
            parsed_prediction,
            raise_on_error=False,
        )
    )


def score_answer(
    completion: Any, solution: Any, verifier_type: Any, choice_options: Any = None
) -> float:
    """根据样本验证类型计算确定性奖励。

    逻辑：先规范化completion和solution，并用extract_required_answer_segment强制
    唯一末行`##### {answer}`；格式不合法立即返回0。合法答案段中，choice类型把
    标准答案归一化为裸选项字母：字母正确得1，唯一匹配正确选项文本得0.5；其余已声明的数值、分数、
    百分比及符号表达类型交给verify_math，未知类型直接报错。

    参数：
        completion：单个模型回答，支持completion_text接受的结构。
        solution：样本的标准最终答案。
        verifier_type：formatter写出的验证类型名称。

    返回：
        格式合法且选择题字母正确为1.0、仅选项内容正确为0.5；否则为0.0。
    """
    expected = str(solution).strip()
    verifier = str(verifier_type).strip().casefold()
    math_verifiers = {
        "math",
        "integer",
        "decimal",
        "fraction",
        "percentage",
        "ratio",
        "latex_fraction",
        "symbolic_number",
        "numeric_expression",
    }
    if verifier != "choice" and verifier not in math_verifiers:
        raise ValueError("未知verifier_type: {}".format(verifier_type))

    prediction = completion_text(completion)
    answer_segment = extract_required_answer_segment(prediction)
    if answer_segment is None:
        return 0.0
    if verifier == "choice":
        return score_choice(answer_segment, expected, choice_options)
    return float(verify_math(answer_segment, expected))


def _as_sequence(value: Any, expected_length: int, name: str) -> Sequence[Any]:
    """把标量监督字段扩展成与completion数量一致的序列。

    逻辑：列表或元组必须与期望长度完全一致；标量则复制expected_length次，供
    reward批量接口逐项zip处理。

    参数：
        value：单个值或已有列表/元组。
        expected_length：目标序列长度。
        name：长度错误信息中使用的字段名。

    返回：
        与completion数量一致的序列。
    """
    if isinstance(value, (list, tuple)):
        if len(value) != expected_length:
            raise ValueError("{}数量与completions不一致".format(name))
        return value
    return [value] * expected_length
