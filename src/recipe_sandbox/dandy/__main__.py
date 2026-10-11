"""Dandy 命令入口：validate 只读；search、train、evaluate 显式执行对应阶段。"""
import argparse
import json
from pathlib import Path


def main():
    """解析操作与配置路径，延迟加载具体执行模块。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "search", "train", "evaluate"])
    parser.add_argument("--config", required=True, help="运行 YAML，路径相对于 AutoSelection 根目录")
    parser.add_argument("--resume", action="store_true", help="仅恢复同配置的 search")
    parser.add_argument("--checkpoint", help="独立 evaluate 使用的模型路径")
    parser.add_argument("--role", choices=["main", "ood"], help="独立 evaluate 的用途")
    args = parser.parse_args()
    from .search import load_run, run
    from .benchmarks.spec import load_benchmarks
    from .data.pool import FixedPool
    if args.command == "search":
        run(args.config, resume=args.resume)
        return
    if args.resume:
        parser.error("--resume 仅适用于 search")
    config, training = load_run(args.config, purpose=args.command)
    if args.command != "train":
        main_benchmarks, ood = load_benchmarks(config["benchmarks"])
    if args.command == "validate":
        pool = FixedPool.from_config(training)
        print(json.dumps({"mode": config["mode"], "pool_size": len(pool.samples),
                          "main": [b.name for b in main_benchmarks], "ood": [b.name for b in ood],
                          "validation": "data/config only; hardware not checked"}, ensure_ascii=False))
    elif args.command == "train":
        from .training.runner import train
        pool = FixedPool.from_config(training)
        root = Path(config["output_dir"])
        root.mkdir(parents=True, exist_ok=False)
        dataset = pool.materialize(pool.records, root / "selected.jsonl")
        checkpoint = train(config["mode"], training, config["runtime"], dataset, root / "model", root / "training.log")
        print(checkpoint)
    else:
        if not args.checkpoint or not args.role:
            parser.error("evaluate 需要 --checkpoint 和 --role")
        from .evaluation.run import evaluate
        from .config import repository_path
        evaluate(repository_path(args.checkpoint), main_benchmarks if args.role == "main" else ood,
                 config["evaluation"], config["runtime"], config["output_dir"])


if __name__ == "__main__":
    main()
