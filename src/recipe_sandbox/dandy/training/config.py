"""统一独立训练阶段及Qwen2.5-VL 7B到3B的OPD-GKD配置契约。

读取阶段YAML；OPD默认从SFT配置继承数据源及抽样设置，不继承训练超参。
静态检查不加载训练依赖；运行前检查本地教师/学生架构、tokenizer与视觉编码
兼容性，防止把不对齐的token分布送入GKD。只读配置和模型元数据，不修改权重。
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping

from ..config import load_yaml, repository_path

TRAINING_STAGES = frozenset({"sft", "grpo", "opd-gkd"})
DATA_FIELDS = ("datasource", "total_sample", "sample_type", "seed")


def load_training_config(stage: str, path: str | Path) -> dict[str, Any]:
    """读取阶段配置；OPD缺省数据字段从dataset_config（默认SFT）深拷贝补齐。

    显式数据字段优先，null保留其原始语义。dataset_config保留为来源快照，
    不传给Swift；引用以仓库根为基准，无递归继承、不创建文件。
    """
    if stage not in TRAINING_STAGES:
        raise ValueError("未知训练阶段: " + stage)
    values = load_yaml(path)
    if stage == "opd-gkd":
        source = values.get("dataset_config", "src/recipe_sandbox/dandy/configs/sft.yaml")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("opd-gkd.dataset_config必须是SFT数据配置路径")
        source = str(repository_path(source))
        data = load_yaml(source)
        for key in DATA_FIELDS:
            if key not in values:
                if key not in data:
                    raise ValueError("SFT数据配置缺少字段: " + key)
                values[key] = copy.deepcopy(data[key])
        values["dataset_config"] = source
    return values


def validate_opd_config(values: Mapping[str, Any]) -> None:
    """验证纯在线反向KL配置，不读取模型文件，非法或混入GRPO参数时立即报错。

    本适配只支持本地冻结教师、完整词表GKD及原生学生生成。限制会明确报错，
    不静默回退至离线蒸馏、top-k近似或GRPO；数值训练参数仍由公共验证器检查。
    """
    fixed = {"lmbda": 1.0, "beta": 1.0, "sft_alpha": 0.0}
    if values.get("train_type") != "full":
        raise ValueError("opd-gkd只支持train_type: full，不适配LoRA/adapter训练")
    for key, expected in fixed.items():
        value = values.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != expected:
            raise ValueError(f"opd-gkd.{key}必须为{expected}")
    for key in ("seq_kd", "use_vllm", "packing", "padding_free", "use_logits_to_keep"):
        if values.get(key) is not False:
            raise ValueError(f"opd-gkd.{key}必须显式为false")
    for key in ("model_path", "teacher_model"):
        if not isinstance(values.get(key), str) or not values[key].strip():
            raise ValueError(f"opd-gkd.{key}必须为本地模型路径")
    for key in ("model_type", "teacher_model_type"):
        if values.get(key, "qwen2_5_vl") != "qwen2_5_vl":
            raise ValueError(f"opd-gkd.{key}只支持qwen2_5_vl")
    if type(values.get("offload_teacher_model", False)) is not bool:
        raise ValueError("opd-gkd.offload_teacher_model必须为布尔值")
    if repository_path(values["model_path"]) == repository_path(values["teacher_model"]):
        raise ValueError("opd-gkd要求7B教师与3B学生使用不同模型目录")
    forbidden = {"reward_funcs", "reward_weights", "dynamic_sample", "max_resample_times",
                 "overlong_filter", "num_generations", "teacher_model_server", "gkd_logits_topk",
                 "teacher_adapters", "ref_model", "ref_adapters", "external_plugins"}
    if forbidden.intersection(values):
        raise ValueError("opd-gkd不接受参数: " + ", ".join(sorted(forbidden.intersection(values))))
    if "rlhf_type" in values and values["rlhf_type"] != "gkd":
        raise ValueError("opd-gkd.rlhf_type只能为gkd")
    for key in ("max_length", "max_completion_length"):
        if type(values.get(key)) is not int or values[key] <= 0:
            raise ValueError(f"opd-gkd.{key}必须为正整数")
    temperature = values.get("temperature")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("opd-gkd.temperature必须为有限正数")


def validate_opd_models(values: Mapping[str, Any]) -> None:
    """运行前检查本地7B教师与3B学生的架构和输入编码，不加载模型权重。

    允许同架构的完整微调checkpoint；按模型config确认规模而非依赖目录名。
    两者词表logits维度允许官方151936/152064差异，由Swift GKD对齐尾部；
    实际token到ID映射、特殊token及视觉patch布局必须一致。不证明NPU可运行。
    """
    validate_opd_config(values)
    roots = [repository_path(values[key]) for key in ("model_path", "teacher_model")]
    configs = [load_yaml(root / "config.json") for root in roots]
    for root, config, size in zip(roots, configs, ((2048, 36, 151936), (3584, 28, 152064))):
        text = config.get("text_config", config)
        actual = (text.get("hidden_size"), text.get("num_hidden_layers"), text.get("vocab_size"))
        if config.get("model_type") != "qwen2_5_vl" or actual != size:
            raise ValueError(f"opd-gkd仅适配Qwen2.5-VL-3B学生/7B教师: {root}")
        if values["max_length"] + values["max_completion_length"] > text.get("max_position_embeddings", 0):
            raise ValueError(f"OPD输入加生成预算超出模型上下文: {root}")
        _validate_weight_files(root)
    for key in ("bos_token_id", "eos_token_id", "image_token_id", "video_token_id",
                "vision_start_token_id", "vision_end_token_id"):
        if configs[0].get(key) is None or configs[0].get(key) != configs[1].get(key):
            raise ValueError("OPD教师/学生特殊token不匹配: " + key)
    for key in ("patch_size", "spatial_merge_size", "temporal_patch_size"):
        a, b = (config.get("vision_config", {}).get(key) for config in configs)
        if a is None or a != b:
            raise ValueError("OPD教师/学生视觉patch不匹配: " + key)
    tokenizers = [json.loads((root / "tokenizer.json").read_text(encoding="utf-8")) for root in roots]
    for key in ("model", "added_tokens", "normalizer", "pre_tokenizer", "post_processor", "decoder"):
        if tokenizers[0].get(key) != tokenizers[1].get(key):
            raise ValueError("OPD教师/学生tokenizer不兼容: " + key)
    if not tokenizers[0].get("model", {}).get("vocab"):
        raise ValueError("OPD tokenizer缺少非空词表")


def _validate_weight_files(root: Path) -> None:
    """确认完整权重文件或索引内全部分片存在；不加载tensor或验证数值内容。"""
    if (root / "model.safetensors").is_file():
        return
    index = root / "model.safetensors.index.json"
    if not index.is_file():
        raise FileNotFoundError(f"OPD需要完整模型权重（非adapter）: {root}")
    weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"OPD权重索引缺少非空weight_map: {index}")
    for name in weight_map.values():
        if not isinstance(name, str) or not name:
            raise ValueError(f"OPD权重分片名称非法: {index}")
        shard = (root / name).resolve()
        if not shard.is_relative_to(root) or not shard.is_file():
            raise FileNotFoundError(f"OPD权重分片缺失或越界: {shard}")
