"""原搜索评测接口到 dandy 训练/图文评测的薄适配；仅接收 main benchmark。"""
from __future__ import annotations

import json
from pathlib import Path
import time

from recipe_sandbox.evaluation.evaluator_base import BaseEvaluator, EvalResult
from .training.runner import train
from .evaluation.run import evaluate as evaluate_benchmarks


class DandyEvaluator(BaseEvaluator):
    """每个候选从共同模型独立训练，checkpoint 保留用于搜索结束后的 OOD。"""

    def __init__(self, pool, main, config, training, root, *, trainer=train, evaluator=evaluate_benchmarks):
        """注入固定池和 main；替身函数只用于无模型测试。"""
        if any(b.role != "main" for b in main):
            raise ValueError("搜索评测器只能接收 main benchmark")
        self.pool, self.main = pool, main
        self.config = {key: config[key] for key in ("mode", "runtime", "evaluation")}
        self.training, self.root = training, Path(root)
        self.trainer, self.evaluator = trainer, evaluator

    def evaluate(self, dataset_path, recipe_name, *, task_names=None, state_vector=None):
        """读取原执行器的 ID 列表，保序物化、训练、汇总 main，返回 0–100 dev_score。"""
        if task_names is not None and set(task_names) != {b.name for b in self.main}:
            raise ValueError("候选评测任务必须与 main 声明一致")
        candidate = self.root / recipe_name
        if candidate.resolve().parent != self.root.resolve():
            raise ValueError("recipe_name 必须是当前运行内的单层目录名")
        ids = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
        candidate.mkdir(parents=True, exist_ok=True)
        id_path = candidate / "ids.json"
        if id_path.exists() and json.loads(id_path.read_text(encoding="utf-8")) != ids:
            raise ValueError("同名候选已有不同 ID 顺序；不能复用训练结果")
        id_path.write_text(json.dumps(ids), encoding="utf-8")
        result_path = candidate / "result.json"
        if result_path.exists():
            return EvalResult(**json.loads(result_path.read_text(encoding="utf-8")))
        dataset = self.pool.materialize(ids, candidate / "selected.jsonl")
        # A failed/interrupted attempt is retained; resume starts a fresh candidate attempt.
        attempt = 1
        while (candidate / f"attempt-{attempt}").exists():
            attempt += 1
        attempt_path = candidate / f"attempt-{attempt}"
        attempt_path.mkdir()
        started = time.perf_counter()
        checkpoint = self.trainer(self.config["mode"], self.training, self.config["runtime"], dataset,
                                  attempt_path / "model", attempt_path / "training.log")
        training_seconds = time.perf_counter() - started
        started = time.perf_counter()
        scores, summary = self.evaluator(checkpoint, self.main, self.config["evaluation"],
                                          self.config["runtime"], attempt_path / "main")
        evaluation_seconds = time.perf_counter() - started
        devices = len(self.config["runtime"]["devices"])
        result = EvalResult(dev_score=100 * sum(scores.values()) / len(scores),
                            train_cost_gpu_hours=training_seconds * devices / 3600,
                            eval_cost_gpu_hours=evaluation_seconds * devices / 3600,
                            task_scores=scores,
                            extra={"checkpoint": str(checkpoint), "summary": summary,
                                   "benchmark_fingerprints": {b.name: b.fingerprint for b in self.main},
                                   "cost_unit": "device_hours", "device_type": self.config["runtime"]["device_type"]})
        (candidate / "result.json").write_text(json.dumps(result.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
