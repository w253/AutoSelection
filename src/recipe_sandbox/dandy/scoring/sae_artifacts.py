"""读取并匹配AutoSelection使用的encoder-only SAE产物契约。

本模块只检查``artifact.json``、``cfg.json``和``sae.safetensors``的静态元信息，
确认产物类型、FP32编码器、mean-nonpadding pooling、hookpoint和输入策略
字段存在。匹配时比较产物内部元信息而不是目录名称；真正加载张量及再次核对权重
形状由后续指标模块调用现有SAE加载接口完成。本模块不导入torch或NPU依赖。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


@dataclass(frozen=True)
class SaeArtifactContract:
    """保存已通过静态契约检查的SAE产物信息。

    ``metadata``来自artifact.json，``configuration``来自cfg.json，``tensor_path``
    指向非空权重文件。对象不表示权重已经加载或真实VLM前向已经验证。
    """

    directory: Path
    metadata: Dict[str, Any]
    configuration: Dict[str, Any]
    tensor_path: Path


def _json_object(path: Path) -> Dict[str, Any]:
    """读取顶层为对象的UTF-8 JSON文件。

    参数：
        path：必须存在的JSON文件。

    返回：
        JSON对象字典；文件缺失、解析失败或顶层非对象时异常向上抛出。
    """
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("SAE元信息顶层必须是对象: {}".format(path))
    return payload


def load_sae_artifact_contract(path: str | Path) -> SaeArtifactContract:
    """检查一个encoder-only SAE目录的静态文件与元信息契约。

    逻辑：要求三个正式文件存在且权重非空，读取两个JSON对象，检查
    hookpoint、输入策略和隐藏维度，并固定要求artifact_type为encoder_only、dtype为
    float32、pooling为mean_nonpadding；同时要求artifact与cfg中的d_in一致。

    参数：
        path：SAE产物目录。

    返回：
        已校验但尚未加载权重的SaeArtifactContract。
    """
    directory = Path(path).resolve()
    if not directory.is_dir():
        raise FileNotFoundError("SAE产物目录不存在: {}".format(directory))
    artifact_path = directory / "artifact.json"
    config_path = directory / "cfg.json"
    tensor_path = directory / "sae.safetensors"
    for required in (artifact_path, config_path, tensor_path):
        if not required.is_file():
            raise FileNotFoundError("SAE产物缺少文件: {}".format(required))
    if tensor_path.stat().st_size <= 0:
        raise ValueError("SAE权重文件为空: {}".format(tensor_path))

    metadata = _json_object(artifact_path)
    configuration = _json_object(config_path)
    expected_values = {
        "artifact_type": "encoder_only",
        "dtype": "float32",
        "pooling": "mean_nonpadding",
    }
    for key, expected in expected_values.items():
        if metadata.get(key) != expected:
            raise ValueError("SAE产物{}必须为{}".format(key, expected))
    if (
        not isinstance(metadata.get("hookpoint"), str)
        or not metadata["hookpoint"].strip()
        or metadata["hookpoint"] != metadata["hookpoint"].strip()
    ):
        raise ValueError("SAE产物缺少hookpoint")
    if not isinstance(metadata.get("input_policy"), dict) or not metadata["input_policy"]:
        raise ValueError("SAE产物缺少input_policy")
    d_in = metadata.get("d_in")
    if isinstance(d_in, bool) or not isinstance(d_in, int) or d_in <= 0:
        raise ValueError("SAE产物d_in必须是正整数")
    config_d_in = configuration.get("d_in")
    if (
        isinstance(config_d_in, bool)
        or not isinstance(config_d_in, int)
        or config_d_in <= 0
        or config_d_in != d_in
    ):
        raise ValueError("SAE artifact.json与cfg.json的d_in不一致")
    return SaeArtifactContract(
        directory=directory,
        metadata=metadata,
        configuration=configuration,
        tensor_path=tensor_path,
    )
