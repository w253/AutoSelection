"""规范化AutoSelection指标运行所需的训练样本与benchmark对话。

本模块把formatter记录转换成统一的``messages``、``images``和回答列表，但不加载
processor、VLM或图片像素。SFT记录直接使用已有assistant消息；GRPO记录在仅含
user prompt时把``solution``追加为唯一assistant回答。benchmark记录只构造问题与
图片，不读取标准答案，确保MONA原型不含答案信息。所有``<image>``占位符均按原始
顺序与图片一一对应，任何数量或角色异常都会在模型前向前失败。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple


@dataclass(frozen=True)
class MetricConversation:
    """保存一条经过严格检查的图文指标对话。

    ``messages``保持formatter原顺序，``images``保存规范化绝对路径，``answers``按
    assistant消息顺序保存回答正文。MONA benchmark对话的``answers``为空；训练池
    对话必须至少有一个回答。
    """

    sample_id: str
    messages: Tuple[Dict[str, str], ...]
    images: Tuple[str, ...]
    answers: Tuple[str, ...]


def _sample_id(record: Mapping[str, Any], location: str) -> str:
    """读取一条记录的稳定非空字符串ID。

    参数：
        record：formatter输出记录。
        location：用于错误信息的数据位置说明。

    返回：
        未改写的稳定ID；字段缺失、类型错误或含首尾空白时抛出ValueError。
    """
    value = record.get("id")
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
    ):
        raise ValueError("{}样本缺少无首尾空白的稳定字符串ID".format(location))
    return value


def _messages(record: Mapping[str, Any], location: str) -> list[Dict[str, str]]:
    """复制并校验formatter消息列表。

    参数：
        record：包含messages字段的记录。
        location：用于错误信息的数据位置说明。

    返回：
        仅含role和content的新字典列表；空消息、未知角色、空正文或多余字段时报错。
    """
    raw_messages = record.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("{}样本必须包含非空messages列表".format(location))
    result = []
    for position, raw in enumerate(raw_messages):
        if not isinstance(raw, Mapping) or set(raw) != {"role", "content"}:
            raise ValueError(
                "{}样本messages第{}项必须只含role和content".format(
                    location, position
                )
            )
        role = raw.get("role")
        content = raw.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(
                "{}样本messages第{}项角色不受支持".format(location, position)
            )
        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                "{}样本messages第{}项content必须是非空字符串".format(
                    location, position
                )
            )
        if "<|im_start|>" in content or "<|im_end|>" in content:
            raise ValueError(
                "{}样本messages第{}项不能包含Qwen角色控制token".format(
                    location, position
                )
            )
        result.append({"role": role, "content": content})
    return result


def _images(record: Mapping[str, Any], location: str) -> Tuple[str, ...]:
    """读取并规范化一条记录的有序图片路径。

    参数：
        record：``images``已经由数据加载层解析的记录。
        location：用于错误信息的数据位置说明。

    返回：
        保持原顺序的绝对图片路径元组；路径缺失或文件不存在时抛出异常。同一图片
        可在一条样本中出现多次，因为模型输入必须忠实保留formatter图片顺序。
    """
    raw_images = record.get("images")
    if not isinstance(raw_images, list) or not raw_images:
        raise ValueError("{}样本必须包含非空images列表".format(location))
    result = []
    for position, raw_path in enumerate(raw_images):
        path = Path(str(raw_path)).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                "{}样本第{}张图片不存在: {}".format(location, position, path)
            )
        key = str(path)
        result.append(key)
    return tuple(result)


def training_metric_conversation(record: Mapping[str, Any]) -> MetricConversation:
    """把SFT或GRPO固定池记录转换成带回答的指标对话。

    逻辑：保留SFT已有的全部assistant消息；若记录没有assistant消息，则要求存在
    非空``solution``并追加一条assistant消息，供GRPO参考答案计算IFD和SAE特征。
    最后核对全部消息中的``<image>``数量与图片数量相等，且assistant正文不含图片
    占位符。

    参数：
        record：已由``datarecipe.data``加载并解析图片路径的固定池记录。

    返回：
        至少含一个回答的MetricConversation，不修改原记录。
    """
    sample_id = _sample_id(record, "训练池")
    messages = _messages(record, "训练池{}".format(sample_id))
    answers = [item["content"] for item in messages if item["role"] == "assistant"]
    if not answers:
        solution = record.get("solution")
        if not isinstance(solution, str) or not solution.strip():
            raise ValueError(
                "训练池{}既无assistant消息也无非空solution".format(sample_id)
            )
        answer = solution
        messages.append({"role": "assistant", "content": answer})
        answers.append(answer)
    images = _images(record, "训练池{}".format(sample_id))
    image_placeholders = sum(
        message["content"].count("<image>") for message in messages
    )
    if image_placeholders != len(images):
        raise ValueError(
            "训练池{}的<image>数量{}与图片数量{}不一致".format(
                sample_id, image_placeholders, len(images)
            )
        )
    if any(
        "<image>" in message["content"]
        for message in messages
        if message["role"] == "assistant"
    ):
        raise ValueError("训练池{}的assistant消息不能包含<image>".format(sample_id))
    return MetricConversation(
        sample_id=sample_id,
        messages=tuple(messages),
        images=images,
        answers=tuple(answers),
    )






def qwen_dialog(conversation: MetricConversation) -> list[Dict[str, Any]]:
    """把规范对话转换成Qwen多模态chat-template内容块。

    逻辑：逐条消息按``<image>``切分，在每个占位符位置插入一个image块，并按
    ``conversation.images``顺序消费图片。返回块中不保存路径，实际PIL图片由运行层
    单独传给processor，避免模板自行重复读取文件。

    参数：
        conversation：已经核对图片占位符数量的MetricConversation。

    返回：
        可传给Qwen processor ``apply_chat_template``的消息列表。
    """
    dialog: list[Dict[str, Any]] = []
    image_position = 0
    for message in conversation.messages:
        content = []
        parts = message["content"].split("<image>")
        for part_position, part in enumerate(parts):
            if part_position:
                if image_position >= len(conversation.images):
                    raise ValueError("Qwen对话图片占位符超过图片数量")
                content.append({"type": "image"})
                image_position += 1
            if part:
                content.append({"type": "text", "text": part})
        dialog.append({"role": message["role"], "content": content})
    if image_position != len(conversation.images):
        raise ValueError("Qwen对话未消费全部图片")
    return dialog


def answer_only_conversation(answer: str, position: int) -> MetricConversation:
    """构造仅含一个assistant回答的文本对话用于IFD分母。

    参数：
        answer：非空回答正文。
        position：原对话中的assistant零基序号，只用于生成内部诊断ID。

    返回：
        无图片、只含assistant消息的MetricConversation；非法正文或序号时报错。
    """
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer-only回答必须是非空字符串")
    if isinstance(position, bool) or not isinstance(position, int) or position < 0:
        raise ValueError("answer-only位置必须是非负整数")
    return MetricConversation(
        sample_id="answer-only-{}".format(position),
        messages=({"role": "assistant", "content": answer},),
        images=(),
        answers=(answer,),
    )
