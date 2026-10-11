"""把多个formatter输出转换成ms-swift可直接加载的混合JSONL。

本模块读取各数据源的data.json或JSONL，校验messages与images字段，将图片路径
转换为绝对路径，再按照配置的总样本数和sample_type执行尽可能均衡的多源取样；
来源容量不足时把未满足的配额转移给仍有余量的来源。最终结果写入训练YAML指定的
dataset_output_path，供SFT或GRPO启动器加载。
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from ..config import repository_path
from ..training.reward import parse_choice_options
from .sampling import formatter_sequential


@dataclass(frozen=True)
class SelectedRecord:
    """关联一条入选训练记录与其配置数据源。

    ``record``保持formatter输出字段及图片顺序，``source_path``保存训练YAML中
    datasource的原始路径字符串，``source_index``保存其零基配置顺序。该包装只供
    数据准备和AutoSelection来源审计使用，不会写入训练JSONL或改动样本正文。
    """

    record: Dict[str, Any]
    source_path: str
    source_index: int


def _data_file(path_value: str) -> Path:
    """定位一个数据源实际使用的数据文件。

    逻辑：先按仓库规则解析路径；目录输入自动追加data.json，文件输入直接使用，
    最后检查目标确实是文件。

    参数：
        path_value：formatter输出目录或具体data.json/JSONL路径。

    返回：
        已存在的数据文件绝对路径。
    """
    path = repository_path(path_value)
    if path.is_dir():
        path = path / "data.json"
    if not path.is_file():
        raise FileNotFoundError("数据文件不存在: {}".format(path))
    return path


def _load_records(path_value: str) -> List[Dict[str, Any]]:
    """读取并规范化一个SFT或RLVR数据源的全部记录。

    逻辑：JSONL按非空行读取，JSON要求顶层data数组；逐条验证字典结构、messages
    和images，并将图片相对路径解析为绝对路径且检查文件存在。

    参数：
        path_value：数据目录、data.json路径或JSONL路径。

    返回：
        保持原始记录顺序、图片字段已绝对路径化的记录列表。
    """
    path = _data_file(path_value)
    if path.suffix.casefold() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as file_object:
            for line_number, line in enumerate(file_object, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("{}第{}行不是字典".format(path, line_number))
                records.append(record)
    else:
        with path.open("r", encoding="utf-8") as file_object:
            payload = json.load(file_object)
        records = payload if isinstance(payload, list) else payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(records, list):
            raise ValueError("{}必须是数组或包含顶层data数组".format(path))

    normalized_records: List[Dict[str, Any]] = []
    for index, source_record in enumerate(records, start=1):
        if not isinstance(source_record, dict):
            raise ValueError("{}第{}条记录不是字典".format(path, index))
        record = dict(source_record)
        messages = record.get("messages")
        images = record.get("images")
        if not isinstance(messages, list) or not messages:
            raise ValueError("{}第{}条记录缺少messages".format(path, index))
        if not isinstance(images, list) or not images:
            raise ValueError("{}第{}条记录缺少images".format(path, index))
        resolved_images = []
        for image in images:
            image_path = Path(str(image))
            if not image_path.is_absolute():
                image_path = path.parent / image_path
            image_path = image_path.resolve()
            if not image_path.is_file():
                raise FileNotFoundError("记录引用的图片不存在: {}".format(image_path))
            resolved_images.append(str(image_path))
        record["images"] = resolved_images
        if str(record.get("verifier_type", "")).casefold() == "choice":
            prompt = "\n".join(
                str(message.get("content", ""))
                for message in messages
                if isinstance(message, dict) and message.get("role") == "user"
            )
            record["choice_options"] = parse_choice_options(prompt)
        normalized_records.append(record)
    return normalized_records


def _allocate_equal_quotas(sizes: Sequence[int], total_sample: int) -> List[int]:
    """将目标样本数尽可能平均且容量感知地分配给多个数据源。

    逻辑：在仍有可用记录的数据源之间反复使用divmod均分剩余配额，余数按数据源
    顺序分配。某个来源达到容量后，将其未满足配额转移给其他仍有余量的来源；仅当
    总需求超过全部来源容量之和时才报错。

    参数：
        sizes：各数据源可用记录数，顺序与datasource一致。
        total_sample：所有来源合并后的目标样本总数。

    返回：
        与sizes等长的整数配额列表；总和等于total_sample且每项不超过对应容量。
    """
    if total_sample <= 0:
        raise ValueError("total_sample必须为正整数或null")
    if not sizes:
        raise ValueError("datasource不能为空")
    if any(size < 0 for size in sizes):
        raise ValueError("数据源容量不能为负数")
    if total_sample > sum(sizes):
        raise ValueError("total_sample超过所有数据源的不重复样本总数")

    quotas = [0 for _ in sizes]
    remaining = total_sample
    active_indices = [index for index, size in enumerate(sizes) if size > 0]
    while remaining:
        base, remainder = divmod(remaining, len(active_indices))
        allocated = 0
        next_active_indices = []
        for position, index in enumerate(active_indices):
            requested = base + int(position < remainder)
            available = sizes[index] - quotas[index]
            selected = min(requested, available)
            quotas[index] += selected
            allocated += selected
            if quotas[index] < sizes[index]:
                next_active_indices.append(index)
        remaining -= allocated
        active_indices = next_active_indices
    return quotas


def select_mixed_records(
    data_sources: Sequence[Dict[str, Any]],
    *,
    total_sample: int | None,
    sample_type: str,
    seed: int,
) -> List[SelectedRecord]:
    """按训练YAML语义确定性选择多个数据源中的记录。

    逻辑：读取所有datasource并计算容量感知的均衡配额，容量不足来源的缺口转移给
    其他来源。sequential对匹配预设路径和字段的来源沿用formatter分层配额与稳定
    顺序，否则回退头部取样；random在各来源内随机抽样，再整体打乱。两种模式都检查
    跨来源非空id重复，并为每条记录保留配置来源信息但不修改记录正文。

    参数：
        data_sources：包含path字段的数据源配置列表。
        total_sample：目标总量；None表示使用各来源全部记录。
        sample_type：取样方式，只允许sequential或random。
        seed：random模式使用的确定性随机种子；sequential模式不消费该随机数。

    返回：
        按最终训练顺序排列的SelectedRecord列表。
    """
    if not data_sources:
        raise ValueError("datasource不能为空")
    if sample_type not in {"sequential", "random"}:
        raise ValueError("sample_type只能是sequential或random")

    source_records: List[List[Dict[str, Any]]] = []
    source_paths: List[str] = []
    for source_index, source in enumerate(data_sources):
        if not isinstance(source, dict) or not source.get("path"):
            raise ValueError(
                "datasource第{}项必须是包含path的字典".format(source_index + 1)
            )
        source_path = str(source["path"])
        source_paths.append(source_path)
        source_records.append(_load_records(source_path))

    if total_sample is None:
        quotas = [len(records) for records in source_records]
    else:
        quotas = _allocate_equal_quotas(
            [len(records) for records in source_records],
            int(total_sample),
        )

    random_generator = random.Random(seed)
    selected_records: List[SelectedRecord] = []
    seen_ids = set()
    for source_index, (source_path, records, quota) in enumerate(
        zip(source_paths, source_records, quotas)
    ):
        if sample_type == "sequential":
            selected = formatter_sequential(_data_file(source_path), records, quota)
        else:
            selected = (
                records
                if quota == len(records)
                else random_generator.sample(records, quota)
            )
        for record in selected:
            record_id = record.get("id")
            if record_id is not None:
                if record_id in seen_ids:
                    raise ValueError("混合数据中出现重复id: {}".format(record_id))
                seen_ids.add(record_id)
            selected_records.append(
                SelectedRecord(
                    record=record,
                    source_path=source_path,
                    source_index=source_index,
                )
            )
    if sample_type == "random":
        random_generator.shuffle(selected_records)
    return selected_records
