"""组装原 AutoSelection 搜索闭环；不复制 GP、LLM 角色或配方执行逻辑。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import yaml

from .config import load_yaml, repository_path, require, api_key
from .training.config import load_training_config, validate_opd_config
from .data.pool import FixedPool
from .benchmarks.spec import load_benchmarks


def load_run(path, *, purpose="search"):
    """读取运行及阶段配置并解析路径；不加载模型、不读取 API 密钥。"""
    config = load_yaml(path)
    mode = config.get("mode") if purpose == "evaluate" else require(config, "mode", "run")
    if purpose != "evaluate" and mode not in {"sft", "grpo", "opd-gkd"}:
        raise ValueError("mode 只能为 sft、grpo、opd-gkd")
    training = {}
    if purpose != "evaluate":
        training = load_training_config(mode, config["stage_configs"][mode])
        for key in ("model_path", "datasource", "sample_type", "seed", "num_train_epochs",
                    "per_device_train_batch_size", "gradient_accumulation_steps", "max_length",
                    "max_pixels", "torch_dtype", "attn_impl"):
            require(training, key, mode)
        # null is an explicit all-record choice, while the template uses 'REQUIRED'.
        total = training["total_sample"]
        if total is not None and (type(total) is not int or total < 1):
            raise ValueError("total_sample 需要显式正整数或 null（全部）")
        training["model_path"] = str(repository_path(training["model_path"]))
        if mode == "opd-gkd":
            training["teacher_model"] = str(repository_path(training["teacher_model"]))
            validate_opd_config(training)
    config["output_dir"] = str(repository_path(require(config, "output_dir", "run")))
    for key in ("device_type", "devices"):
        require(config["runtime"], key, "runtime")
    from .training.runner import runtime_environment
    runtime_environment(config["runtime"])
    if purpose == "train":
        return config, training
    for section in ("engine", "generation"):
        for key in config["evaluation"][section]:
            require(config["evaluation"][section], key, f"evaluation.{section}")
    if type(config["evaluation"]["batch_size"]) is not int or config["evaluation"]["batch_size"] < 1:
        raise ValueError("evaluation.batch_size 必须为显式正整数")
    tp = config["evaluation"]["engine"]["tensor_parallel_size"]
    if type(tp) is not int or not 1 <= tp <= len(config["runtime"]["devices"]):
        raise ValueError("evaluation TP 需为正整数且不超过设备数")
    benchmarks = config["benchmarks"]
    benchmarks["directories"] = [str(repository_path(p)) for p in benchmarks["directories"]]
    benchmarks["paths"] = {k: str(repository_path(v)) for k, v in benchmarks.get("paths", {}).items()}
    if purpose == "evaluate":
        return config, training
    for key in ("base_model", "sae_path"):
        config["scoring"][key] = str(repository_path(require(config["scoring"], key, "scoring")))
    for key in ("device", "sae_token_batch_size", "varentropy_token_chunk_size"):
        require(config["scoring"], key, "scoring")
    for key in ("max_evaluations", "n_lhs_seeds", "stagnation_patience", "min_viable_samples", "k_exploration"):
        require(config["search"], key, "search")
    if config["search"]["max_evaluations"] < 1 or config["search"]["min_viable_samples"] < 0:
        raise ValueError("搜索预算需为正数，最小样本边界需非负")
    for key in ("model", "base_url", "api_key_env"):
        require(config["llm"], key, "llm")
    config["catalog"] = str(repository_path(config["catalog"]))
    return config, training


def build_search(config, training, pool, main, sparse_cache, *, action=None, feedback=None,
                 selection=None, evaluator=None):
    """注入原执行器、状态计算与六算子；可注入替身验证完整调用链。"""
    from recipe_sandbox.pipeline.task_config import TaskConfig, ModelConfig, LLMConfig
    from recipe_sandbox.pipeline.task_manager import TaskManager
    from recipe_sandbox.pipeline.recipe_executor import RecipeExecutor
    from recipe_sandbox.search.mcts_search import MCTSSearchLoop
    from recipe_sandbox.agents.action_llm import ActionLLMGenerator
    from recipe_sandbox.agents.feedback_llm import FeedbackLLM
    from recipe_sandbox.agents.selection_llm import SelectionLLM
    from .operators import build_registry
    from .evaluator import DandyEvaluator

    root = Path(config["output_dir"])
    task = TaskConfig(task_name="dandy", output_dir=str(root),
                      model=ModelConfig(model_path=config["scoring"]["base_model"], d_sae=sparse_cache.d_sae))
    manager = TaskManager(task)
    registry = build_registry(config["operators"])
    executor = RecipeExecutor(task, manager, registry, sparse_cache=sparse_cache,
                              cached_train_samples=pool.samples, isolate_metadata=True,
                              min_viable_samples=config["search"]["min_viable_samples"])
    # Use the existing catalog, filtered for this run; never modify its source file.
    catalog = load_yaml(config["catalog"])
    catalog["families"] = {name: {**family, "operators": {k: v for k, v in family["operators"].items() if k in registry.names()}}
                           for name, family in catalog["families"].items()}
    catalog_path = root / "operator_catalog.yaml"
    catalog_path.write_text(yaml.safe_dump(catalog, allow_unicode=True, sort_keys=False), encoding="utf-8")
    if action is None or feedback is None or selection is None:
        llm = config["llm"]
        from .llm import DandyLLMClient
        key = api_key(llm)
        llm_config = LLMConfig(api_key=key, base_url=llm["base_url"], model=llm["model"])
        action = ActionLLMGenerator(llm_config=llm_config, catalog_path=str(catalog_path), registered_operators=set(registry.names()))
        feedback = FeedbackLLM(llm_config)
        selection = SelectionLLM(llm_config)
        client = DandyLLMClient(llm, key)
        action.client = feedback.client = selection.client = client
    evaluator = evaluator or DandyEvaluator(pool, main, config, training, root / "candidates")
    options = config["search"]
    return MCTSSearchLoop(manager, executor, action, evaluator, str(catalog_path),
                          max_evaluations=options["max_evaluations"], n_lhs_seeds=options["n_lhs_seeds"],
                          stagnation_patience=options["stagnation_patience"], k_exploration=options["k_exploration"],
                          pool_size=len(pool.samples), feedback_llm=feedback, selection_llm=selection,
                          search_log_path=str(root / "search_log.jsonl"))


def restore_search(loop, root):
    """复用原日志恢复并补齐 dandy 评测结果和状态，供 LLM/GP 继续使用。"""
    from recipe_sandbox.evaluation.evaluator_base import EvalResult
    root = Path(root)
    log = root / "search_log.jsonl"
    if not log.exists():
        return
    loop.resume_from_log(str(log))
    entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    for candidate, entry in zip(loop.history, entries):
        candidate.output_samples = entry.get("output_samples", 0)
        result_path = root / "candidates" / candidate.recipe.recipe_name / "result.json"
        if entry.get("evaluation_completed"):
            candidate.eval_result = EvalResult(**json.loads(result_path.read_text(encoding="utf-8")))
            if not Path(candidate.eval_result.extra["checkpoint"]).is_dir():
                raise FileNotFoundError("恢复搜索所需 checkpoint 已丢失")
        trace = root / "recipes" / candidate.recipe.recipe_name / "trace.json"
        if trace.exists():
            saved = json.loads(trace.read_text(encoding="utf-8"))
            candidate.step_traces = saved["steps"]
    loop._update_surrogate()


def run(path, *, resume=False):
    """计算固定池指标后调用原搜索；只对最终 main 最佳模型评测 OOD。"""
    from .scoring.cache_adapter import prepare_metrics
    from .evaluation.run import evaluate

    config, training = load_run(path)
    # Fail before expensive forward passes if the chosen credential is absent.
    if not api_key(config["llm"]):
        raise ValueError("LLM API key 未在环境变量或根目录 .env 中设置")
    main, ood = load_benchmarks(config["benchmarks"])
    pool = FixedPool.from_config(training)
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "dandy_run.json"
    identity_values = {"config": config, "training": training,
                       "main": {b.name: b.fingerprint for b in main},
                       "ood": {b.name: b.fingerprint for b in ood}}
    identity = hashlib.sha256(json.dumps(identity_values, sort_keys=True).encode()).hexdigest()
    if resume:
        if json.loads(manifest.read_text(encoding="utf-8"))["identity"] != identity:
            raise ValueError("运行配置或 benchmark 已变，不能恢复旧搜索")
    else:
        if manifest.exists() or (root / "search_log.jsonl").exists():
            raise FileExistsError("运行目录已使用；使用 --resume 或新的 output_dir")
        manifest.write_text(json.dumps({"identity": identity, **identity_values}, ensure_ascii=False, indent=2), encoding="utf-8")
    cache = prepare_metrics(pool, main, config["scoring"], training, root / "metrics")
    loop = build_search(config, training, pool, main, cache)
    if resume:
        restore_search(loop, root)
    loop.search()
    completed = [candidate for candidate in loop.history if candidate.eval_result is not None]
    if not completed:
        raise RuntimeError("搜索没有已完成的训练评测结果")
    best = max(completed, key=lambda candidate: candidate.score)
    checkpoint = best.eval_result.extra["checkpoint"]
    ood_path = root / "ood" / "summary.json"
    if ood and not ood_path.exists():
        evaluate(checkpoint, ood, config["evaluation"], config["runtime"], root / "ood")
    (root / "best.json").write_text(json.dumps({"recipe": best.recipe.recipe_name, "dev_score": best.score,
                                               "checkpoint": checkpoint}, indent=2), encoding="utf-8")
    return best
