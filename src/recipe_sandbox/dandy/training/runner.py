"""共享 SWIFT 训练入口：迁入阶段参数与临时数据逻辑，不探测 HPC 设备。"""
from __future__ import annotations
import importlib.metadata
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping
import yaml
from ..config import REPOSITORY_ROOT, repository_path, require
from .config import TRAINING_STAGES, validate_opd_config, validate_opd_models
from .data import read_training_jsonl
from .process_runtime import run_managed_process
PROJECT_ONLY_KEYS = {"datasource", "dataset_output_path", "model_path", "sample_type",
                     "total_sample", "train_type", "dataset_config"}


def runtime_environment(runtime):
    """使用显式设备列表和进程数；不探测设备、不自动选择并行策略。"""
    environment = dict(os.environ)
    device_type = runtime["device_type"]
    devices = runtime["devices"]
    if device_type not in {"npu", "cuda", "cpu"} or not isinstance(devices, list) or not devices:
        raise ValueError("runtime 需要 device_type 和非空 devices 列表")
    if len(set(devices)) != len(devices):
        raise ValueError("runtime.devices 不能重复")
    if device_type in {"npu", "cuda"}:
        key = "ASCEND_RT_VISIBLE_DEVICES" if device_type == "npu" else "CUDA_VISIBLE_DEVICES"
        environment[key] = ",".join(str(device) for device in devices)
    environment["NPROC_PER_NODE"] = str(len(devices))
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    return environment


def train(stage, config, runtime, dataset_path, output_dir, log_path):
    """从共同起点执行一次候选训练，返回实际最终 checkpoint。

    参数均由调用方提供；只覆盖候选输出与最终 checkpoint 保存方式。临时训练视图
    按已有字段白名单生成，成功或失败都会删除；所有训练模式复用本入口。
    """
    values = dict(config)
    if stage == "opd-gkd":
        validate_opd_models(values)
    environment = runtime_environment(runtime)
    batch = int(values["per_device_train_batch_size"]) * int(values["gradient_accumulation_steps"]) * len(runtime["devices"])
    if stage == "grpo" and batch % int(values["num_generations"]):
        raise ValueError("GRPO 全局 completion batch 必须能被 num_generations 整除")
    if stage == "grpo" and values.get("use_vllm") and values.get("vllm_mode") == "colocate":
        tp = int(values["vllm_tensor_parallel_size"])
        if tp < 1 or len(runtime["devices"]) % tp:
            raise ValueError("GRPO rollout TP 必须整除显式设备数")
    if stage == "grpo" and runtime["device_type"] == "npu":
        environment["VLLM_ASCEND_ENABLE_NZ"] = "0"
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    values.update(output_dir=str(output), save_strategy="steps", save_steps=1_000_000_000,
                  save_total_limit=1, save_only_model=True, add_version=False)
    text = read_training_jsonl(Path(dataset_path), stage)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".jsonl",
                                         prefix=".swift-input-", dir=Path(dataset_path).parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        arguments = _swift_arguments(stage, values, temporary)
        _run_swift("sft" if stage == "sft" else "rlhf", arguments, environment, log_path=log_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    checkpoints = sorted((p for p in output.glob("checkpoint-*") if p.is_dir() and p.name[11:].isdigit()),
                         key=lambda p: int(p.name[11:]))
    checkpoint = checkpoints[-1] if checkpoints else output
    if not (checkpoint / "config.json").is_file() or not (list(checkpoint.glob("*.safetensors")) or list(checkpoint.glob("pytorch_model*.bin"))):
        raise FileNotFoundError(f"训练未产生可评测的完整 checkpoint: {checkpoint}")
    return checkpoint
def _ms_swift_major_version() -> int:
    """读取当前环境安装的ms-swift主版本号。

    逻辑：查询ms-swift发行包元数据并提取版本字符串第一个整数，用于选择3.x的
    train_type或4.x的tuner_type；包不存在或版本格式异常时给出明确错误。

    参数：
        无。

    返回：
        ms-swift主版本号整数。
    """
    try:
        version = importlib.metadata.version("ms-swift")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("当前容器中未安装ms-swift") from error
    try:
        return int(version.split(".", 1)[0])
    except ValueError as error:
        raise RuntimeError("无法识别ms-swift版本: {}".format(version)) from error

def _swift_arguments(
    stage: str,
    config: Dict[str, Any],
    dataset_path: Path,
) -> Dict[str, Any]:
    """把项目训练配置转换成ms-swift可接受的参数字典。

    逻辑：移除datasource等项目专用字段，拒绝其余空值，解析模型、数据和输出路径，
    按ms-swift主版本转换训练类型。GRPO还会注入rlhf_type、自定义reward名称和
    reward.py插件绝对路径；浮点奖励权重仅在传给CLI的副本中转为文本，以适配
    ms-swift 4.x的列表参数展开与命令打印，最终由其训练参数解析器恢复为浮点值。

    参数：
        stage：sft、grpo或opd-gkd训练阶段。
        config：阶段YAML的完整配置字典。
        dataset_path：已经生成的混合JSONL绝对路径。

    返回：
        可序列化为ms-swift临时YAML的参数字典。
    """
    if stage not in TRAINING_STAGES:
        raise ValueError("未知训练阶段: " + stage)
    if stage == "opd-gkd":
        validate_opd_config(config)
    arguments = {
        key: value
        for key, value in config.items()
        if key not in PROJECT_ONLY_KEYS
    }
    null_keys = sorted(key for key, value in arguments.items() if value is None)
    if null_keys:
        raise ValueError(
            "{}配置仍有未填写参数: {}".format(stage.upper(), ", ".join(null_keys))
        )

    model_path = repository_path(require(config, "model_path", "{}配置".format(stage.upper())))
    arguments["model"] = str(model_path)
    arguments["dataset"] = str(dataset_path)

    train_type = str(require(config, "train_type", "{}配置".format(stage.upper())))
    if stage == "opd-gkd":
        from packaging.version import Version
        if Version(importlib.metadata.version("ms-swift")) < Version("4.0.1"):
            raise RuntimeError("opd-gkd适配要求ms-swift>=4.0.1；请先核对HPC依赖")
    if _ms_swift_major_version() >= 4:
        arguments["tuner_type"] = train_type
    else:
        arguments["train_type"] = train_type

    arguments["output_dir"] = str(repository_path(require(arguments, "output_dir", "训练配置")))
    if stage == "grpo":
        arguments["rlhf_type"] = "grpo"
        reward_functions = list(arguments.get("reward_funcs") or [])
        if "datarecipe_accuracy" not in reward_functions:
            reward_functions.append("datarecipe_accuracy")
        arguments["reward_funcs"] = reward_functions
        # ms-swift 4.x会把配置中的列表值展开为CLI参数，再直接join(args)打印命令。
        # 浮点权重必须先转为文本；训练参数解析器随后仍将其恢复为float。
        if "reward_weights" in arguments:
            arguments["reward_weights"] = [
                str(weight) for weight in arguments["reward_weights"]
            ]
        arguments["external_plugins"] = [
            str(Path(__file__).with_name("reward_plugin.py"))
        ]
        if arguments.get("dynamic_sample"):
            arguments["external_plugins"].append(
                str(Path(__file__).with_name("grpo_resample_patch.py"))
            )
    if stage == "opd-gkd":
        arguments["rlhf_type"] = "gkd"
        arguments["teacher_model"] = str(repository_path(config["teacher_model"]))
    return arguments

def _run_swift(
    command_name: str,
    arguments: Dict[str, Any],
    environment: Dict[str, str],
    *,
    log_path: str | Path | None = None,
) -> None:
    """通过临时YAML执行一个ms-swift训练子进程。

    逻辑：打印最终预算（正max_steps会覆盖epoch），把参数字典安全写入临时YAML，
    用--config把该YAML交给swift阶段子命令，
    并传入已经构建的NPU环境。无论进程成功或失败，finally都会删除临时YAML。

    参数：
        command_name：ms-swift子命令，当前为sft或rlhf。
        arguments：已经转换完成的ms-swift参数字典。
        environment：包含NPU可见设备设置的完整进程环境。
        log_path：可选合并标准输出与错误的日志文件；缺省继承当前终端。

    返回：
        无；子进程失败时由subprocess抛出CalledProcessError。
    """
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            encoding="utf-8",
            delete=False,
        ) as temporary_file:
            yaml.safe_dump(
                arguments,
                temporary_file,
                allow_unicode=True,
                sort_keys=False,
            )
            temporary_path = Path(temporary_file.name)
        budget = {key: arguments.get(key) for key in (
            "tuner_type", "train_type", "num_train_epochs", "max_steps", "learning_rate",
            "warmup_ratio", "warmup_steps", "per_device_train_batch_size",
            "gradient_accumulation_steps", "deepspeed", "gradient_checkpointing")}
        budget_text = "[DataRecipe effective training budget] " + str(budget)
        print(budget_text, flush=True)
        if (arguments.get("max_steps") or -1) > 0:
            print("[DataRecipe] max_steps > 0：本次为步数预算，覆盖epoch停止条件", flush=True)
        # ms-swift 4.x只把位置参数里的.json当配置解析，YAML必须走--config。
        command = ["swift", command_name, "--config", str(temporary_path)]
        if log_path is None:
            run_managed_process(
                command,
                cwd=REPOSITORY_ROOT,
                env=environment,
            )
        else:
            destination = Path(log_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x", encoding="utf-8") as log_stream:
                log_stream.write(budget_text + "\n")
                log_stream.flush()
                run_managed_process(
                    command,
                    cwd=REPOSITORY_ROOT,
                    env=environment,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
