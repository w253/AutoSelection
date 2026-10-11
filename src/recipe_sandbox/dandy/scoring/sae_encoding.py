"""把冻结VLM激活编码为AutoSelection使用的样本级Top-192 SAE特征。

本模块复用DataRecipe已验证的encoder-only SAE加载/编码接口，但不复用外部
``AutoSelection/``实现。调用方负责用匹配产物的基础VLM提取全部非padding token
激活；本模块逐token编码、对样本求均值、选择最大的正激活并返回缓存行。
Torch与sparsify依赖只在实际调用时加载，普通配置和状态检查不触发NPU初始化。
"""

from __future__ import annotations

from .records import SparseFeatureRow


SAE_SAMPLE_TOP_K = 192


def encode_sample_sae_feature(
    encoder,
    activations,
    *,
    sample_id: str,
    token_batch_size: int = 256,
) -> SparseFeatureRow:
    """生成一条样本均值池化后的Top-192正激活稀疏向量。

    逻辑：调用现有``sample_feature``逐token执行FP32 encoder并求均值，不进行解码；
    再从完整SAE字典选择最多192个最大值并删除非正项。Top-K是算法常量，不接受
    YAML覆盖。

    参数：
        encoder：由DataRecipe SAE导出接口加载且处于目标设备的encoder-only模型。
        activations：形状``[nonpadding_tokens, d_in]``的冻结VLM激活Torch张量。
        sample_id：formatter稳定样本ID。
        token_batch_size：SAE逐token编码时的正整数分块大小，不改变数值定义。

    返回：
        可直接写入运行级Safetensors缓存的SparseFeatureRow；没有正激活时抛出
        ValueError，不使用零向量占位。
    """
    import torch

    from .sae_loader import sample_feature

    if (
        not isinstance(sample_id, str)
        or not sample_id
        or sample_id != sample_id.strip()
    ):
        raise ValueError("SAE特征sample_id必须是无首尾空白的非空字符串")
    feature = sample_feature(
        encoder,
        activations,
        token_batch_size=token_batch_size,
    )
    if feature.ndim != 1 or feature.numel() <= 0:
        raise ValueError("SAE样本均值特征必须是一维非空张量")
    count = min(SAE_SAMPLE_TOP_K, int(feature.numel()))
    values, indices = torch.topk(feature.float(), k=count, largest=True, sorted=True)
    positive = values > 0
    values = values[positive]
    indices = indices[positive]
    if values.numel() == 0 or not bool(torch.isfinite(values).all().item()):
        raise ValueError("SAE样本特征没有有限正激活")
    return SparseFeatureRow(
        sample_id=sample_id,
        indices=tuple(int(index) for index in indices.detach().cpu().tolist()),
        values=tuple(float(value) for value in values.detach().cpu().tolist()),
    )
