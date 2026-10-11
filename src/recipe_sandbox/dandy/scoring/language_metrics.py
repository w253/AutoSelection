"""计算VLM assistant-token NLL与Varentropy充分统计量。

本模块只处理已经由模型前向得到且与目标token对齐的logits，不负责加载VLM、构造
chat template、定位assistant区间或选择NPU。调用方必须显式提供目标token和布尔
assistant mask，从而让条件回答与仅答案两次前向使用同一套可审计公式。Torch在函数
调用时才导入，配置校验和纯数学状态聚合不会因此加载NPU运行库。
"""

from __future__ import annotations

from .records import MeanStatistic


def _validate_aligned_tensors(logits, target_ids, assistant_mask) -> None:
    """校验logits、目标token和assistant mask具有明确对齐关系。

    参数：
        logits：形状为``[positions, vocabulary]``的浮点Torch张量。
        target_ids：形状为``[positions]``的整数Torch张量，每项是同位置预测目标。
        assistant_mask：形状为``[positions]``的bool Torch张量。

    返回：
        无；设备、形状、dtype、词表边界或有效assistant位置非法时抛出ValueError。
    """
    import torch

    if not all(isinstance(value, torch.Tensor) for value in (logits, target_ids, assistant_mask)):
        raise TypeError("语言指标输入必须全部是Torch张量")
    if logits.ndim != 2 or logits.shape[0] <= 0 or logits.shape[1] <= 1:
        raise ValueError("logits必须是非空[positions, vocabulary]二维张量")
    if target_ids.ndim != 1 or assistant_mask.ndim != 1:
        raise ValueError("target_ids和assistant_mask必须是一维张量")
    if target_ids.shape[0] != logits.shape[0] or assistant_mask.shape[0] != logits.shape[0]:
        raise ValueError("语言指标张量的positions维度不一致")
    if logits.device != target_ids.device or logits.device != assistant_mask.device:
        raise ValueError("语言指标张量必须位于同一设备")
    if not logits.dtype.is_floating_point:
        raise ValueError("logits必须是浮点张量")
    if target_ids.dtype == torch.bool or target_ids.dtype.is_floating_point:
        raise ValueError("target_ids必须是整数张量")
    if assistant_mask.dtype != torch.bool:
        raise ValueError("assistant_mask必须是bool张量")
    selected_targets = target_ids[assistant_mask]
    if selected_targets.numel() == 0:
        raise ValueError("assistant_mask没有选中任何目标token")
    if bool(((selected_targets < 0) | (selected_targets >= logits.shape[1])).any().item()):
        raise ValueError("assistant目标token超出logits词表范围")


def assistant_nll_from_logits(logits, target_ids, assistant_mask) -> MeanStatistic:
    """计算assistant目标token的负对数似然总和与计数。

    逻辑：只选择mask为真的位置，将logits转为FP32，用``logsumexp-logit[target]``
    直接计算每个目标token的NLL，避免保存完整softmax。函数适用于图文条件回答前向
    和仅答案前向；两者分别调用后再由``SampleMetricRecord.ifd_score``求平均NLL比值。

    参数：
        logits：与预测目标对齐的``[positions, vocabulary]``张量。
        target_ids：每个position的目标token ID。
        assistant_mask：标明哪些position属于回答目标。

    返回：
        CPU浮点总和与assistant token数量组成的MeanStatistic；非有限结果时报错。
    """
    import torch

    _validate_aligned_tensors(logits, target_ids, assistant_mask)
    selected_logits = logits[assistant_mask].float()
    selected_targets = target_ids[assistant_mask].long()
    target_logits = selected_logits.gather(1, selected_targets[:, None]).squeeze(1)
    nll = torch.logsumexp(selected_logits, dim=1) - target_logits
    if not bool(torch.isfinite(nll).all().item()) or bool((nll < 0).any().item()):
        raise ValueError("assistant NLL包含非有限数或负数")
    return MeanStatistic(float(nll.double().sum().item()), int(nll.numel()))


def assistant_varentropy_from_logits(
    logits,
    target_ids,
    assistant_mask,
    *,
    token_chunk_size: int = 32,
) -> MeanStatistic:
    """计算assistant位置完整词表surprisal分布的Varentropy。

    逻辑：对每个assistant位置计算完整词表``p``与``s=-log(p)``，先求熵
    ``H=sum(p*s)``，再求``sum(p*(s-H)^2)``；最终返回各assistant位置Varentropy
    的总和与计数。按token分块可限制临时FP32 log-probability张量大小，但不会截断
    词表、抽样词表或改写定义。

    参数：
        logits：与预测目标对齐的``[positions, vocabulary]``张量。
        target_ids：仅用于复用严格对齐和词表边界检查的目标token ID。
        assistant_mask：标明回答目标位置的bool张量。
        token_chunk_size：每次转换为完整FP32 log-probability的正整数位置数。

    返回：
        Varentropy总和与assistant位置数；非有限结果或非法批量时报错。
    """
    import torch

    _validate_aligned_tensors(logits, target_ids, assistant_mask)
    if (
        isinstance(token_chunk_size, bool)
        or not isinstance(token_chunk_size, int)
        or token_chunk_size <= 0
    ):
        raise ValueError("token_chunk_size必须是正整数")
    selected_logits = logits[assistant_mask]
    total = torch.zeros((), dtype=torch.float64, device="cpu")
    for chunk in selected_logits.split(token_chunk_size):
        log_probabilities = torch.log_softmax(chunk.float(), dim=-1)
        probabilities = log_probabilities.exp()
        surprisals = -log_probabilities
        entropy = (probabilities * surprisals).sum(dim=-1, keepdim=True)
        values = (probabilities * (surprisals - entropy).square()).sum(dim=-1)
        if not bool(torch.isfinite(values).all().item()) or bool((values < 0).any().item()):
            raise ValueError("assistant Varentropy包含非有限数或负数")
        total += values.detach().double().cpu().sum()
    return MeanStatistic(float(total.item()), int(selected_logits.shape[0]))
