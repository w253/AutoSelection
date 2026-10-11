"""单个隔离进程加载 SWIFT/vLLM，一次处理全部指定 benchmark。

使用显式 TP 与设备列表，不迁入 HPC 自动分片调度。请求不包含答案，结果按 ID
写回。父进程负责 main/OOD 选择、判分和进程清理。
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

from ..config import REPOSITORY_ROOT
from ..training.runner import runtime_environment
from ..training.process_runtime import run_managed_process
from .scorers import _response_text


def infer(checkpoint, requests, config, runtime, directory):
    """启动单次隔离推理；请求及预测文件留在该次运行目录供核对。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    request_path = directory / "inference.json"
    output_path = directory / "predictions.json"
    payload = {"checkpoint": str(checkpoint), "requests": requests, "config": config, "output": str(output_path)}
    request_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tp = config["engine"]["tensor_parallel_size"]
    if not isinstance(tp, int) or tp < 1 or tp > len(runtime["devices"]):
        raise ValueError("推理 TP 需要显式正整数且不得超过设备数")
    environment = runtime_environment(runtime)
    environment.pop("NPROC_PER_NODE", None)
    with (directory / "inference.log").open("w", encoding="utf-8") as stream:
        run_managed_process([sys.executable, "-m", __name__, str(request_path)],
                            cwd=REPOSITORY_ROOT, env=environment, stdout=stream, stderr=-2, text=True)
    return json.loads(output_path.read_text(encoding="utf-8"))


def worker(path):
    """仅在子进程中导入 SWIFT；同步 infer 保持请求顺序并核对数量。"""
    from swift import InferRequest, RequestConfig
    from swift.infer_engine import VllmEngine
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    config = values["config"]
    engine = VllmEngine(str(values["checkpoint"]), **config["engine"])
    request_config = RequestConfig(**config["generation"])
    requests = values["requests"]
    result = []
    size = config["batch_size"]
    if not isinstance(size, int) or size < 1:
        raise ValueError("推理 batch_size 必须为正整数")
    for start in range(0, len(requests), size):
        batch = requests[start:start + size]
        outputs = engine.infer([InferRequest(**r["input"]) for r in batch], request_config=request_config)
        if len(outputs) != len(batch):
            raise ValueError("推理输出数量不匹配")
        result.extend({"benchmark": r["benchmark"], "id": r["id"], "prediction": _response_text(o)} for r, o in zip(batch, outputs))
    Path(values["output"]).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    worker(sys.argv[1])
