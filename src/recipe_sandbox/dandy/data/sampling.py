"""在已格式化候选池内复现预设formatter的sequential配额，不导入转换器重依赖。

仅匹配仓库预设路径、稳定ID及必要分层字段；不匹配返回原有头部取样。
配额基于当前格式化池，不恢复已被formatter过滤或未导出的原始记录。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import re

from ..config import REPOSITORY_ROOT

PRESETS = {
    "sft/LLaVA-CoT-100k": ("LLaVA-CoT-100k", ("source",)),
    "sft/SciEGQA-Train": ("SciEGQA-Train", ("category", "question_type")),
    "sft/VisualWebInstruct-verified": ("VisualWebInstruct-verified", ("batch",)),
    "rl/Vision-R1-rl": ("Vision-R1-rl", ("verifier_type",)),
    "rl/MMK12": ("MMK12-RV", ()),
    "rl/multimodal-open-r1-8k-verified": ("multimodal-open-r1-8k-verified", ()),
}


def _proportional(counts: dict, total: int) -> dict:
    """输入分组容量和目标量，以formatter最大余数法返回精确整数配额。"""
    size = sum(counts.values())
    if not total:
        return dict.fromkeys(counts, 0)
    quotas = {key: count * total // size for key, count in counts.items()}
    order = sorted(counts, key=lambda key: (-(counts[key] * total % size), key))
    for key in order[:total - sum(quotas.values())]:
        quotas[key] += 1
    return quotas


def _difficulty_quotas(counts: dict, total: int) -> dict:
    """按VisualWeb的5:30:40:20:5权重迭代处理容量饱和，返回各难度配额。"""
    weights = {1: 5, 2: 30, 3: 40, 4: 20, 5: 5}
    quotas = dict.fromkeys(weights, 0)
    active = {key for key in weights if counts.get(key, 0)}
    remaining = total
    while remaining:
        weight = sum(weights[key] for key in active)
        raw = {key: remaining * weights[key] / weight for key in active}
        saturated = {key for key in active if counts[key] - quotas[key] < raw[key]}
        if saturated:
            for key in sorted(saturated):
                remaining -= counts[key] - quotas[key]
                quotas[key] = counts[key]
            active -= saturated
            continue
        for key in active:
            quotas[key] += int(raw[key])
        remaining -= sum(int(value) for value in raw.values())
        for key in sorted(active, key=lambda key: (-(raw[key] - int(raw[key])), key))[:remaining]:
            quotas[key] += 1
        remaining = 0
    return quotas


def formatter_sequential(data_file: Path, records: list[dict], quota: int) -> list[dict]:
    """输入真实数据文件、记录与源内配额，返回formatter兼容的稳定有序子集。

    已知路径还须符合ID前缀和公开字段约定；缺字段或自定义数据打印回退原因，
    使用records[:quota]。全量/零配额不改变顺序；random由调用方保留旧行为。
    """
    if quota == 0 or quota == len(records):
        return records[:quota]
    preset = next((key for key in PRESETS if data_file.resolve() in {
        (REPOSITORY_ROOT / "datasets" / key / "formatted_data" / name).resolve()
        for name in ("data.json", "data.jsonl")
    }), None)
    reason = "不是预设数据路径"
    if preset:
        prefix, fields = PRESETS[preset]
        compatible = all(
            re.fullmatch(re.escape(prefix) + r"-\d+", str(row.get("id", "")))
            and all(isinstance(row.get(key), str) and row[key] for key in fields)
            for row in records
        )
        if preset == "sft/VisualWebInstruct-verified":
            compatible = compatible and all(type(row.get("difficulty")) is int
                and row["difficulty"] in range(1, 6)
                and isinstance(row.get("images"), list) and row["images"] for row in records)
        if preset == "rl/Vision-R1-rl":
            compatible = compatible and all(row.get("verifier_type") in {"choice", "math"} for row in records)
        reason = "ID或分层字段不符合formatter输出（VisualWeb旧文件缺batch时需重新格式化）"
        if compatible and len({row["id"] for row in records}) != len(records):
            raise ValueError(f"预设数据中存在重复id: {data_file}")
        if compatible:
            ordered = sorted(records, key=lambda row: int(row["id"].rsplit("-", 1)[1]))
            groups = defaultdict(list)
            for row in ordered:
                key = tuple(row[field] for field in fields)
                if preset == "sft/VisualWebInstruct-verified":
                    key = (row["difficulty"], row["batch"], len(row["images"]))
                groups[key].append(row)
            counts = {key: len(group) for key, group in groups.items()}
            if preset == "sft/VisualWebInstruct-verified":
                by_difficulty = Counter(row["difficulty"] for row in ordered)
                targets = _difficulty_quotas(by_difficulty, quota)
                quotas = {}
                for difficulty, target in targets.items():
                    quotas.update(_proportional({key: count for key, count in counts.items()
                                                if key[0] == difficulty}, target))
            elif preset == "rl/Vision-R1-rl":
                choice = round(quota * counts.get(("choice",), 0) / len(records))
                quotas = {("choice",): choice, ("math",): quota - choice}
            else:
                quotas = _proportional(counts, quota)
            selected = {row["id"] for key, group in groups.items() for row in group[:quotas[key]]}
            print(f"[sampling] {preset}: formatter sequential，{len(selected)}/{len(records)}", flush=True)
            return [row for row in ordered if row["id"] in selected]
    print(f"[sampling] {data_file}: 回退头部取样，原因：{reason}", flush=True)
    return records[:quota]
