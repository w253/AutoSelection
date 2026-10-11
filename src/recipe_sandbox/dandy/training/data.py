"""把完整formatter记录投影为字段和类型固定的ms-swift训练输入。

原记录保留在选样快照或固定池中；本模块只复制训练所需字段，不改写问题、答案、
图片和顺序。SFT/OPD-GKD只输出id/messages/images；GRPO额外输出监督字段，选项映射使用
JSON字符串，避免不同题目的选项字母被Arrow推断成不同struct。依赖基础配置和标准库，
不在数据准备时额外触发datasets或设备后端加载。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import TRAINING_STAGES


def training_record(record: Mapping[str, Any], stage: str) -> dict[str, Any]:
    """验证单条原记录并返回独立的训练字段副本。

    输入stage只能为sft/grpo/opd-gkd；消息使用项目formatter的文本content格式。GRPO
    必须提供非空字符串solution和reward支持的验证类型，缺省选项写为字符串{}。
    类型不合法立即抛出ValueError，不隐式转换答案或丢弃样本。
    """
    if stage not in TRAINING_STAGES:
        raise ValueError("stage只能是sft、grpo或opd-gkd")
    if not isinstance(record, Mapping):
        raise ValueError("训练记录必须是对象")
    sample_id = record.get("id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError("训练记录id必须是非空字符串")
    messages = record.get("messages")
    images = record.get("images")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{sample_id}: messages必须是非空列表")
    copied_messages = []
    for message in messages:
        if (not isinstance(message, dict)
                or not isinstance(message.get("role"), str)
                or not message["role"].strip()
                or not isinstance(message.get("content"), str)):
            raise ValueError(f"{sample_id}: 消息role/content必须是字符串")
        copied_messages.append({"role": message["role"], "content": message["content"]})
    if (not isinstance(images, list) or not images
            or any(not isinstance(item, str) or not item.strip() for item in images)):
        raise ValueError(f"{sample_id}: images必须是非空图片路径字符串列表")
    result = {"id": sample_id, "messages": copied_messages, "images": list(images)}
    if stage == "grpo":
        solution = record.get("solution")
        verifier = record.get("verifier_type")
        if not isinstance(solution, str) or not solution.strip():
            raise ValueError(f"{sample_id}: GRPO solution必须是非空字符串")
        if not isinstance(verifier, str) or verifier.strip().casefold() not in {
                "choice", "math", "integer", "decimal", "fraction", "percentage",
                "ratio", "latex_fraction", "symbolic_number", "numeric_expression"}:
            raise ValueError(f"{sample_id}: GRPO verifier_type不受reward支持")
        options = record.get("choice_options")
        if options is None:
            options = {}
        if isinstance(options, str):
            options = json.loads(options)
        if (not isinstance(options, dict)
                or any(not isinstance(k, str) or not isinstance(v, str) for k, v in options.items())):
            raise ValueError(f"{sample_id}: choice_options必须是字符串映射或其JSON字符串")
        result.update(solution=solution, verifier_type=verifier,
                      choice_options=json.dumps(options, ensure_ascii=False, allow_nan=False))
    return result


def training_jsonl(records: Iterable[Mapping[str, Any]], stage: str) -> str:
    """按输入顺序生成统一JSONL文本，拒绝空集、重复ID及非法记录。

    不修改传入记录；先完成全部验证再返回文本，使调用方不会因中途字段错误写出
    半个训练文件。每条记录的顶层及消息字段固定，GRPO选项始终为字符串。
    """
    lines = []
    seen = set()
    for position, record in enumerate(records, 1):
        try:
            row = training_record(record, stage)
        except (ValueError, TypeError) as error:
            raise ValueError(f"第{position}条训练记录非法: {error}") from error
        if row["id"] in seen:
            raise ValueError(f"训练记录id重复: {row['id']}")
        seen.add(row["id"])
        lines.append(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    if not lines:
        raise ValueError("训练数据不能为空")
    return "".join(lines)


def read_training_jsonl(path: Path, stage: str) -> str:
    """读取已有JSONL并返回统一训练文本，不覆盖原文件、不重新取样。

    输入路径必须存在；空行忽略，解析及字段错误在训练子进程启动前抛出。
    用于公共训练入口接收旧快照时再次保证字段契约。
    """
    with path.open(encoding="utf-8") as stream:
        return training_jsonl((json.loads(line) for line in stream if line.strip()), stage)
