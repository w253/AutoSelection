"""声明式图文 benchmark：加载、分区隔离和指纹，不构建通用插件框架。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import yaml


def resolve_callable(reference):
    """显式 module:function 引用，仅在实际使用时导入。"""
    module, name = reference.split(":", 1)
    return getattr(importlib.import_module(module), name)


@dataclass(frozen=True)
class Benchmark:
    """已加载的声明与两种分区；不把 OOD 对象传给搜索评测器。"""

    name: str
    role: str
    spec: dict
    data: tuple
    prototype: tuple
    fingerprint: str


def _records(rows, root):
    """保留 benchmark 记录，解析图片路径，并校验唯一 ID。"""
    result, seen = [], set()
    for source in rows:
        row = dict(source)
        sample_id = row["id"]
        if not isinstance(sample_id, str) or not sample_id.strip() or sample_id in seen:
            raise ValueError("benchmark ID 必须为唯一的非空字符串")
        seen.add(sample_id)
        images = row.get("images") or ([row["img_path"]] if row.get("img_path") else [])
        if not images:
            raise ValueError(f"benchmark {sample_id} 缺少图片")
        row["images"] = [str((root / image).resolve()) for image in images]
        for image in row["images"]:
            if not Path(image).is_file():
                raise FileNotFoundError(image)
        result.append(row)
    return tuple(result)


def load_benchmarks(config):
    """发现任意声明目录；只对启用的 main/ood 读取数据并校验隔离。"""
    declarations = {}
    for directory in config["directories"]:
        for path in sorted(Path(directory).rglob("benchmark.yaml")):
            if "_template" in path.parts:
                continue
            spec = yaml.safe_load(path.read_text(encoding="utf-8"))
            name = spec["name"]
            if name in declarations:
                raise ValueError(f"重复 benchmark 声明: {name}")
            declarations[name] = (path, spec)
    main, ood = config["main"], config["ood"]
    if len(set(main + ood)) != len(main + ood):
        raise ValueError("main/ood 不能重复或重叠")
    groups = {"main": [], "ood": []}
    for role, names in (("main", main), ("ood", ood)):
        for name in names:
            declaration, original = declarations[name]
            spec = dict(original)
            for field in ("version", "label", "params"):
                spec[field]
            if role not in spec["roles"]:
                raise ValueError(f"{name} 不允许用作 {role}")
            if role == "ood" and "prototype" in spec["partitions"]:
                raise ValueError(f"OOD {name} 不能声明原型分区")
            data_path = Path(config.get("paths", {}).get(name, declaration.parent / spec["data_path"]))
            payload = json.loads(data_path.read_text(encoding="utf-8"))
            data = _records(payload[spec["partitions"]["eval"]], data_path.parent)
            prototype = _records(payload[spec["partitions"]["prototype"]], data_path.parent) if role == "main" and "prototype" in spec["partitions"] else ()
            if not data or (set(r["id"] for r in data) & set(r["id"] for r in prototype)):
                raise ValueError(f"{name} 评测为空或与原型 ID 重叠")
            # Only pure prompt/scoring functions are resolved; no inference backend imports.
            resolve_callable(spec["scorer"])
            resolve_callable(spec["message_builder"])
            if spec["scorer"] == "recipe_sandbox.dandy.evaluation.scorers:answer_types":
                from ..evaluation.scorers import _score_type
                weight = spec["params"]["invalid_format_fallback_weight"]
                if not isinstance(weight, (int, float)) or not 0 <= weight <= 1:
                    raise ValueError("格式回退权重必须在 [0,1]")
                for row in data:
                    _score_type(row)
            content = json.dumps({"spec": spec, "data": data, "prototype": prototype}, sort_keys=True, ensure_ascii=False)
            fingerprint = hashlib.sha256(content.encode()).hexdigest()
            groups[role].append(Benchmark(name, role, spec, data, prototype, fingerprint))
    if not groups["main"]:
        raise ValueError("至少需要一个 main benchmark")
    return groups["main"], groups["ood"]
