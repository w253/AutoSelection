"""声明驱动的评测与汇总：预测按 ID 对齐，main/OOD 由调用方显式分开。"""
from __future__ import annotations

import json
import math
from pathlib import Path

from ..benchmarks.spec import resolve_callable
from .inference import infer


def evaluate(checkpoint, benchmarks, config, runtime, directory, *, inference=infer):
    """评测给定的一组 benchmark；返回 0–1 的任务分数与结果路径。"""
    if not benchmarks:
        raise ValueError("没有启用的评测 benchmark")
    requests = []
    for benchmark in benchmarks:
        builder = resolve_callable(benchmark.spec["message_builder"])
        params = benchmark.spec["params"]
        for record in benchmark.data:
            requests.append({"benchmark": benchmark.name, "id": record["id"], "input": builder(record, params)})
    predictions = inference(checkpoint, requests, config, runtime, directory)
    keys = [(p["benchmark"], p["id"]) for p in predictions]
    expected = {(r["benchmark"], r["id"]) for r in requests}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("预测 ID 缺失、重复或多出")
    by_id = {key: row["prediction"] for key, row in zip(keys, predictions)}
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    scores, summaries = {}, {}
    for benchmark in benchmarks:
        scorer = resolve_callable(benchmark.spec["scorer"])
        rows = []
        for record in benchmark.data:
            prediction = by_id[(benchmark.name, record["id"])]
            scored = scorer(prediction, record, benchmark.spec["params"])
            if not math.isfinite(scored["score"]) or not 0 <= scored["score"] <= 1:
                raise ValueError("benchmark scorer 必须返回有限的 0–1 score")
            rows.append({"id": record["id"], "prediction": prediction, "answer": record["answer"], **scored})
        score = sum(row["score"] for row in rows) / len(rows)
        scores[benchmark.name] = score
        summaries[benchmark.name] = {"score": score, "count": len(rows), "label": benchmark.spec["label"],
                                      "fingerprint": benchmark.fingerprint, "role": benchmark.role}
        (root / f"{benchmark.name}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    return scores, str(root / "summary.json")
