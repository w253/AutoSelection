"""迁入已有 SAE 的只读加载与编码功能；不包含训练、导出和产物备份。"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from sparsify import Sae, SaeConfig

def model_identity(path: Path) -> dict:
    """读取本地模型配置和文件标记，不对大型权重计算完整哈希。

    输入为现有模型目录，返回配置及文件名、大小、修改时间，
    包括 tokenizer、processor 和自定义代码文件。缺少配置时抛出文件读写错误，
    缺少权重时抛出 ValueError。文件标记用于低成本、保守的变更检查，
    不等同于密码学内容校验。"""
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    files = sorted(list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin")))
    if not files:
        raise ValueError(f"no local model weights found: {path}")
    auxiliary = sorted({file for pattern in ["*.json", "*.py", "*.txt", "*.model", "*.jinja"] for file in path.glob(pattern)})
    return {"config": config, "weights": [{"name": file.name, "size": file.stat().st_size,
                                           "mtime_ns": file.stat().st_mtime_ns} for file in files],
            "auxiliary_files": [{"name": file.name, "size": file.stat().st_size,
                                  "mtime_ns": file.stat().st_mtime_ns} for file in auxiliary]}


def load_encoder(directory: str | Path, model_path: str | Path, hookpoint: str,
                 input_policy: dict, device: str = "cpu") -> Sae:
    """核对模型、层和预处理信息后，加载 FP32 编码器。

    输入指定模型产物、预期本地 VLM、层路径及元信息中保存的输入策略。
    返回评估模式的 sparsify 编码器，不包含解码器；
    绑定信息不匹配或权重无法识别时抛出 ValueError 或 RuntimeError。"""
    directory = Path(directory)
    info = json.loads((directory / "artifact.json").read_text(encoding="utf-8"))
    if info["model_identity"] != model_identity(Path(model_path)) or info["hookpoint"] != hookpoint:
        raise ValueError("encoder belongs to a different model checkpoint or layer")
    if info["input_policy"] != input_policy or info["artifact_type"] != "encoder_only":
        raise ValueError("encoder input/preprocessing policy mismatch")
    cfg = json.loads((directory / "cfg.json").read_text(encoding="utf-8"))
    d_in = cfg.pop("d_in")
    encoder = Sae(d_in, SaeConfig.from_dict(cfg), device=device, dtype=torch.float32, decoder=False)
    state = load_file(str(directory / "sae.safetensors"), device="cpu")
    if set(state) != {"encoder.weight", "encoder.bias", "b_dec"} or any(t.dtype != torch.float32 for t in state.values()):
        raise ValueError("expected three FP32 inference tensors")
    encoder.load_state_dict(state, strict=True)
    return encoder.eval()


@torch.no_grad()
def sample_feature(encoder: Sae, activations: torch.Tensor, token_batch_size: int = 256) -> torch.Tensor:
    """逐 token 编码后平均全部非填充位置，不执行解码。

    输入为形状 (tokens, d_in) 的激活和正整数编码批量上限，
    返回位于 encoder.device 的 FP32 隐变量向量。
    使用稀疏 scatter-add 累加，避免构建逐 token 的密集编码矩阵。
    输入为空或包含非有限数值时抛出 ValueError。"""
    if activations.ndim != 2 or activations.shape[1] != encoder.d_in or not len(activations) or token_batch_size < 1:
        raise ValueError("invalid token activations or batch size")
    if not torch.isfinite(activations).all():
        raise ValueError("nonfinite activations")
    feature = torch.zeros(encoder.num_latents, device=encoder.device, dtype=torch.float32)
    for chunk in activations.split(token_batch_size):
        encoded = encoder.encode(chunk.to(encoder.device, dtype=torch.float32))
        feature.scatter_add_(0, encoded.top_indices.flatten(), encoded.top_acts.float().flatten())
    return feature / len(activations)
