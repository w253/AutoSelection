"""Dandy 配置与路径：只读取配置，不探测硬件或更改用户环境。"""
from __future__ import annotations

import json
from pathlib import Path
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def repository_path(value):
    """解析仓库相对路径；绝对路径可引用仓库外的数据与模型。"""
    path = Path(value)
    return (path if path.is_absolute() else REPOSITORY_ROOT / path).resolve()


def load_yaml(path):
    """读取 YAML/JSON 对象，不展开密钥或修改配置。"""
    path = repository_path(path)
    with path.open(encoding="utf-8") as stream:
        return json.load(stream) if path.suffix == ".json" else yaml.safe_load(stream)


def require(config, key, source):
    """要求运行所需字段显式填写，避免把模板当作已确认实验。"""
    value = config[key]
    if value is None:
        raise ValueError(f"{source}.{key} 尚未填写")
    return value
