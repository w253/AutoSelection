"""把图文前向结果接到原 MONA/稀疏缓存；不实现第二套状态统计。

仅完整结果可复用。缓存身份绑定原记录、图片文件标记、模型文件标记、SAE、
main 原型与前向配置；不包含 OOD。超长训练样本同步剔除，不补抽。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

from recipe_sandbox.scoring.sparse_features import SparseFeatureCache, compute_mona_score_arrays
from .metric_inputs import MetricConversation, training_metric_conversation
from .sae_artifacts import load_sae_artifact_contract
from .vlm_runtime import QwenSaeFeatureRuntime, QwenLanguageMetricRuntime, SequenceTooLongError

VERSION = "dandy-assistant-v1"


def file_identity(path):
    """记录模型或产物文件标记；与原 SAE 契约一样，不宣称是权重内容哈希。"""
    root = Path(path)
    return [{"name": str(p.relative_to(root)), "size": p.stat().st_size,
             "mtime_ns": p.stat().st_mtime_ns}
            for p in sorted(root.rglob("*")) if p.is_file()]


def cache_identity(pool, main, config, training):
    """计算本次前向身份；只包含 main 原型，不把 OOD 信息带入指标。"""
    images = sorted({p for r in pool.records.values() for p in r["images"]} |
                    {p for b in main for r in b.prototype for p in r["images"]})
    payload = {"version": VERSION, "records": pool.records, "pool_ids": list(pool.records),
               "images": [(p, Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in images],
               "main_prototypes": {b.name: [{k: r[k] for k in ("id", "images", "text_input")} for r in b.prototype] for b in main}, "config": config,
               "training": {k: training[k] for k in ("model_path", "torch_dtype", "max_pixels", "max_length", "attn_impl")},
               "completion": training.get("max_completion_length"),
               "base_identity": file_identity(config["base_model"]),
               "stage_identity": file_identity(training["model_path"]),
               "sae_identity": file_identity(config["sae_path"])}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def prepare_metrics(pool, main, config, training, directory, *, sae_factory=QwenSaeFeatureRuntime,
                    language_factory=QwenLanguageMetricRuntime):
    """计算或恢复完整有效池；工厂参数供无硬件接口测试使用。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    key = cache_identity(pool, main, config, training)
    complete = directory / "complete.json"
    rows_path = directory / "metrics.jsonl"
    artifact = load_sae_artifact_contract(config["sae_path"])
    d_sae = artifact.configuration.get("num_latents") or artifact.configuration["d_in"] * artifact.configuration["expansion_factor"]
    if complete.exists():
        saved = json.loads(complete.read_text(encoding="utf-8"))
        if saved["identity"] != key:
            raise ValueError("指标缓存身份不匹配；请使用新的运行目录")
        raw = rows_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != saved["rows_sha256"]:
            raise ValueError("指标缓存不完整或已改变")
        rows = [json.loads(line) for line in raw.decode().splitlines()]
        if [r["id"] for r in rows] != saved["ids"]:
            raise ValueError("指标缓存 ID 顺序不匹配")
    else:
        sparse, rejected, targets = {}, {}, {}
        runtime = sae_factory(model_path=config["base_model"], artifact=artifact,
                              device=config["device"], dtype=training["torch_dtype"],
                              token_batch_size=config["sae_token_batch_size"])
        try:
            for sample_id, record in pool.records.items():
                try:
                    feature = runtime.encode(training_metric_conversation(record))
                    sparse[sample_id] = {"indices": list(feature.indices), "values": list(feature.values)}
                except SequenceTooLongError as error:
                    rejected[sample_id] = {"phase": "sae", "length": error.length, "limit": error.limit}
            for benchmark in main:
                if not benchmark.prototype:
                    raise ValueError(f"MONA 需要 {benchmark.name} 的独立原型分区")
                target = np.zeros(d_sae, dtype=np.float32)
                for record in benchmark.prototype:
                    prompt = "\n".join("<image>" for _ in record["images"]) + "\n\n" + record["text_input"]
                    conversation = MetricConversation(record["id"], ({"role": "user", "content": prompt},), tuple(record["images"]), ())
                    feature = runtime.encode(conversation)
                    np.add.at(target, list(feature.indices), feature.values)
                targets[benchmark.name] = target / len(benchmark.prototype)
        finally:
            runtime.close()
        runtime = language_factory(model_path=training["model_path"], stage_config=training,
                                   device=config["device"], varentropy_token_chunk_size=config["varentropy_token_chunk_size"])
        rows = []
        try:
            for sample_id, record in pool.records.items():
                if sample_id not in sparse:
                    continue
                try:
                    result = runtime.compute(training_metric_conversation(record))
                except SequenceTooLongError as error:
                    rejected[sample_id] = {"phase": "language", "length": error.length, "limit": error.limit}
                    continue
                if result.answer_only_nll.mean <= 0:
                    raise ValueError(f"IFD answer-only NLL 非正: {sample_id}")
                rows.append({"id": sample_id, "extra": {
                    "sae_topk": sparse[sample_id],
                    "ifd": {"score": result.conditional_nll.mean / result.answer_only_nll.mean},
                    "varentropy": {"score": result.varentropy.mean},
                }})
        finally:
            runtime.close()
        pool.retain(r["id"] for r in rows)
        for sample, row in zip(pool.samples, rows):
            sample.metadata.extra.update(row["extra"])
        scores = compute_mona_score_arrays(pool.samples, targets)
        for i, row in enumerate(rows):
            row["extra"]["mona_scores"] = {name: float(values[i]) for name, values in scores.items()}
        raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        rows_path.write_bytes(raw)
        complete.write_text(json.dumps({"identity": key, "ids": [r["id"] for r in rows],
                                       "rows_sha256": hashlib.sha256(raw).hexdigest(),
                                       "rejected": rejected}, ensure_ascii=False, indent=2), encoding="utf-8")
    pool.retain(r["id"] for r in rows)
    if not pool.samples:
        raise ValueError("指标过滤后固定池为空")
    by_id = {row["id"]: row["extra"] for row in rows}
    for sample in pool.samples:
        sample.metadata.extra.update(by_id[sample.sample_id])
    return SparseFeatureCache.from_samples(pool.samples, d_sae=d_sae)
