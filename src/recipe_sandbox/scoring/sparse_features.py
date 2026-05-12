"""SAE Sparse Feature Caching + SNAR Computation.

Design:

1. **Ingest Phase (once)**: Run SAE on all N samples. For each sample,
   store ONLY the top-K active feature indices + values as a compact
   sparse dict in ``metadata.extra["sae_topk"]``:

       {"indices": [12, 45, 789, ...], "values": [0.82, 0.34, 0.12, ...]}

   This is ~192 ints + 192 floats = ~1.5 KB per sample vs ~65536 floats
   (~256 KB) for a dense SAE feature vector. Memory savings: ~170x.

2. **Search Loop (fast CPU ops)**: When a Recipe filters N→M samples,
   we reconstruct aggregate statistics from the cached sparse features
   using pure NumPy/CPU operations:

   - **SNAR** (Sparse Neuron Activation Rate): for each feature dim,
     count how many of the M samples have it active.
     SNAR = active_count / M → a (D_SAE,) vector of activation rates.

   - **Distribution Drift**: L2 distance between SNAR(reference) and
     SNAR(filtered), normalized by √D_SAE.

   - **Source Entropy (SAE-based)**: entropy of the aggregate activation
     distribution across feature dims.

   All of these are O(M * K) where K=192, so for M=30K samples:
   30,000 * 192 = ~6M ops → **milliseconds** on CPU.

3. **Integration**: ``DataStateComputer`` uses the cached sparse features
   for ``distribution_drift`` when available.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from recipe_sandbox.schema.types import CanonicalSample

logger = logging.getLogger(__name__)

# Default top-K features to cache per sample
DEFAULT_TOP_K = 192

# Key in metadata.extra where sparse features are stored
SPARSE_KEY = "sae_topk"

# Key in metadata.extra where IFD scores are stored
IFD_KEY = "ifd"


# -----------------------------------------------------------------------
#  IFD (Instruction-Following Difficulty) Computation
#  Piggybacked on SAE ingest forward pass to avoid extra GPU passes.
#
#  IFD(sample) = loss(response | instruction) / loss(response)
#  - loss(response | instruction): CE loss on response tokens from the
#    full (instruction + response) forward pass (already computed for SAE)
#  - loss(response): CE loss from a response-only forward pass
#
#  Reference: Cherry LLM (2024) — "From Quantity to Quality"
# -----------------------------------------------------------------------

def _extract_response_text(sample: CanonicalSample) -> str:
    """Extract the last assistant response text from a sample."""
    from recipe_sandbox.schema.enums import Role
    for msg in reversed(sample.messages):
        if msg.role == Role.ASSISTANT and msg.content.strip():
            return msg.content.strip()
    # Fallback: use target text
    if sample.target.text:
        return sample.target.text.strip()
    return ""


def _compute_ifd_for_batch(
    model,
    tokenizer,
    full_texts: list,
    response_texts: list,
    device: str,
    max_length: int = 2048,
) -> list:
    """Compute IFD scores for a batch using the model.

    This runs TWO forward passes:
    1. Full text (instruction + response) → loss on response tokens
    2. Response only → loss on response tokens

    Returns list of float IFD scores (one per sample).
    """
    import torch
    import torch.nn.functional as F

    def _batch_loss(texts: list) -> list:
        """Compute per-sample mean CE loss for a list of texts."""
        if not texts:
            return []
        inputs = tokenizer(
            texts, return_tensors="pt", truncation=True,
            max_length=max_length, padding=True,
        )
        for k, v in inputs.items():
            if hasattr(v, "to"):
                if k in ["input_ids", "attention_mask", "position_ids"]:
                    inputs[k] = v.to(device=torch.device(device), dtype=torch.long)
                else:
                    inputs[k] = v.to(device)

        with torch.inference_mode():
            outputs = model(**inputs)
        logits = outputs.logits  # (B, seq_len, vocab)

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = inputs["input_ids"][:, 1:].contiguous()
        attention = inputs["attention_mask"][:, 1:].contiguous()

        # Per-token CE loss
        loss_per_token = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_labels.shape)

        # Token-level Varentropy calculation
        probs = F.softmax(shift_logits, dim=-1)
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_entropy = -torch.sum(probs * log_probs, dim=-1)
        token_varentropy = torch.sum(probs * (log_probs + token_entropy.unsqueeze(-1))**2, dim=-1)

        # Mask padding tokens and compute per-sample mean
        valid_lengths = attention.sum(dim=1).clamp(min=1)
        
        masked_loss = loss_per_token * attention.float()
        per_sample_loss = masked_loss.sum(dim=1) / valid_lengths
        
        masked_ve = token_varentropy * attention.float()
        per_sample_ve = masked_ve.sum(dim=1) / valid_lengths
        
        return per_sample_loss.cpu().tolist(), per_sample_ve.cpu().tolist()

    # 1. Loss on full text (instruction + response)
    full_losses, full_ve = _batch_loss(full_texts)
    # 2. Loss on response-only text
    resp_losses, resp_ve = _batch_loss(response_texts)

    # 3. IFD = loss(response | instruction) / loss(response)
    ifd_scores = []
    varentropy_scores = []
    for fl, rl, r_ve in zip(full_losses, resp_losses, resp_ve):
        varentropy_scores.append(r_ve)
        if rl > 1e-8:
            ifd_scores.append(fl / rl)
        else:
            ifd_scores.append(1.0)  # No response → neutral score
    return ifd_scores, varentropy_scores


# -----------------------------------------------------------------------
#  Sparse Feature Storage
# -----------------------------------------------------------------------

def sparse_from_dense(
    dense_feature: Any,  # torch.Tensor or list/np.ndarray, shape (D_SAE,)
    top_k: int = DEFAULT_TOP_K,
) -> Dict[str, list]:
    """Convert a dense SAE feature vector to sparse top-K representation.

    Args:
        dense_feature: 1D feature vector of shape (D_SAE,).
        top_k: Number of top active features to keep.

    Returns:
        {"indices": [int, ...], "values": [float, ...]} — sorted by index.
    """
    arr = _to_numpy(dense_feature)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D feature, got shape {arr.shape}")

    # Get top-K by absolute magnitude
    k = min(top_k, len(arr))
    if k == 0:
        return {"indices": [], "values": []}

    # Use argpartition for O(N) instead of O(N log N) full sort
    abs_arr = np.abs(arr)
    top_indices = np.argpartition(abs_arr, -k)[-k:]

    # Sort by index for consistent ordering
    top_indices = np.sort(top_indices)
    top_values = arr[top_indices]

    # Filter out zeros
    nonzero_mask = top_values != 0.0
    top_indices = top_indices[nonzero_mask]
    top_values = top_values[nonzero_mask]

    return {
        "indices": top_indices.tolist(),
        "values": top_values.tolist(),
    }


def sparse_from_sae_output(
    top_acts: Any,  # (K,) tensor of activation values
    top_indices: Any,  # (K,) tensor of feature indices
) -> Dict[str, list]:
    """Convert SAE sparse output directly (no densification needed).

    This is used when the SAE already outputs (top_acts, top_indices).
    """
    acts = _to_numpy(top_acts).flatten()
    idxs = _to_numpy(top_indices).flatten().astype(int)

    # Filter zeros
    mask = acts != 0.0
    return {
        "indices": idxs[mask].tolist(),
        "values": acts[mask].tolist(),
    }


def annotate_sample_sparse(
    sample: CanonicalSample,
    dense_feature: Any,
    top_k: int = DEFAULT_TOP_K,
) -> None:
    """Attach sparse top-K SAE features to a sample's metadata."""
    sparse = sparse_from_dense(dense_feature, top_k=top_k)
    sample.metadata.extra[SPARSE_KEY] = sparse


def get_sparse_features(sample: CanonicalSample) -> Optional[Dict[str, list]]:
    """Get cached sparse features from a sample, if present."""
    return sample.metadata.extra.get(SPARSE_KEY)


def has_sparse_features(sample: CanonicalSample) -> bool:
    """Check if a sample has cached sparse features."""
    topk = sample.metadata.extra.get(SPARSE_KEY)
    return topk is not None and bool(topk.get("indices"))


# -----------------------------------------------------------------------
#  SNAR (Sparse Neuron Activation Rate)
# -----------------------------------------------------------------------

def compute_snar(
    samples: Sequence[CanonicalSample],
    d_sae: int,
) -> np.ndarray:
    """Compute SNAR (Sparse Neuron Activation Rate) from cached sparse features.

    SNAR[j] = (number of samples where feature j is active) / N

    Returns:
        np.ndarray of shape (d_sae,) — activation rate per feature dim.

    Complexity: O(N * K) where K=avg active features per sample (~192).
    """
    n = len(samples)
    if n == 0:
        return np.zeros(d_sae, dtype=np.float32)

    valid_indices: list[np.ndarray] = []
    valid = 0

    for sample in samples:
        topk = get_sparse_features(sample)
        if topk is None:
            continue
        valid += 1
        indices = np.asarray(topk["indices"], dtype=np.int32)
        if indices.size > 0:
            valid_indices.append(indices)

    if valid == 0:
        return np.zeros(d_sae, dtype=np.float32)

    if not valid_indices:
        return np.zeros(d_sae, dtype=np.float32)

    flat_indices = np.concatenate(valid_indices)
    flat_indices = flat_indices[(flat_indices >= 0) & (flat_indices < d_sae)]
    if flat_indices.size == 0:
        return np.zeros(d_sae, dtype=np.float32)

    counts = np.bincount(flat_indices, minlength=d_sae).astype(np.float32, copy=False)
    return counts / float(valid)


def compute_mean_activation(
    samples: Sequence[CanonicalSample],
    d_sae: int,
) -> np.ndarray:
    """Compute mean activation vector from cached sparse features.

    mean_act[j] = mean of activation values at feature j across samples.

    Returns:
        np.ndarray of shape (d_sae,) — mean activation per feature dim.
    """
    n = len(samples)
    if n == 0:
        return np.zeros(d_sae, dtype=np.float32)

    valid_indices: list[np.ndarray] = []
    valid_values: list[np.ndarray] = []
    valid = 0

    for sample in samples:
        topk = get_sparse_features(sample)
        if topk is None:
            continue
        valid += 1
        indices = np.asarray(topk["indices"], dtype=np.int32)
        values = np.asarray(topk["values"], dtype=np.float64)
        if indices.size == 0 or values.size == 0:
            continue
        size = min(indices.size, values.size)
        if size == 0:
            continue
        valid_indices.append(indices[:size])
        valid_values.append(values[:size])

    if valid == 0:
        return np.zeros(d_sae, dtype=np.float32)

    if not valid_indices:
        return np.zeros(d_sae, dtype=np.float32)

    flat_indices = np.concatenate(valid_indices)
    flat_values = np.concatenate(valid_values)
    mask = (flat_indices >= 0) & (flat_indices < d_sae)
    if not np.any(mask):
        return np.zeros(d_sae, dtype=np.float32)
    sums = np.bincount(
        flat_indices[mask],
        weights=flat_values[mask],
        minlength=d_sae,
    )
    return (sums / float(valid)).astype(np.float32, copy=False)


# -----------------------------------------------------------------------
#  Fast CPU-based Drift & Entropy from Sparse Features
# -----------------------------------------------------------------------

def sparse_distribution_drift(
    reference_snar: np.ndarray,
    current_snar: np.ndarray,
) -> float:
    """L2 distance between two SNAR vectors, normalized by √D.

    This measures how much the activation pattern of the filtered dataset
    has drifted from the reference dataset.

    Returns:
        float in [0, ∞), typically [0, 1] for well-behaved data.
    """
    d = len(reference_snar)
    if d == 0:
        return 0.0
    diff = current_snar - reference_snar
    l2 = float(np.sqrt(np.sum(diff ** 2)))
    return l2 / math.sqrt(d)


def sparse_activation_entropy(snar: np.ndarray) -> float:
    """Shannon entropy of the SNAR distribution.

    Treats SNAR as a probability distribution (after normalization).
    High entropy = features are activated uniformly across many dimensions.
    Low entropy = activations concentrate on few dimensions.

    Returns:
        float — entropy in nats.
    """
    total = snar.sum()
    if total <= 0:
        return 0.0
    p = snar / total
    # Filter zeros for log
    mask = p > 0
    return float(-np.sum(p[mask] * np.log(p[mask])))


def sparse_generalized_jaccard(
    mean_act_a: np.ndarray,
    mean_act_b: np.ndarray,
) -> float:
    """Generalized Jaccard similarity between two mean activation vectors.

    This is the same metric as MONA scoring but computed on aggregates.
    """
    intersection = np.minimum(mean_act_a, mean_act_b).sum()
    union = np.maximum(mean_act_a, mean_act_b).sum()
    if union <= 0:
        return 0.0
    return float(intersection / union)


def sparse_jaccard_score(
    sample: CanonicalSample,
    target_vector: np.ndarray,
) -> float:
    """Compute MONA-style score: jaccard(sample_sparse, eval_target_vector).

    Uses the cached ``sae_topk`` sparse representation — no GPU needed.
    The target vector is the mean activation of eval-set samples.

    This mirrors ``generalized_jaccard_similarity()`` in ``mona.py``
    but operates on the sparse cache: only touches K=192 dims per sample.

    Returns:
        float in [0, 1] — higher means more relevant to the eval task.
    """
    topk = get_sparse_features(sample)
    if not topk or not topk["indices"]:
        return 0.0

    indices = topk["indices"]
    values = topk["values"]

    # Compute min/max only at the sample's active indices
    # For dims not active in the sample:
    #   min(0, target[j]) = 0 (no contribution to intersection)
    #   max(0, target[j]) = target[j] (adds to union)
    sample_sum_min = 0.0
    sample_sum_max = 0.0
    for idx, val in zip(indices, values):
        t = float(target_vector[idx])
        sample_sum_min += min(val, t)
        sample_sum_max += max(val, t)

    # For dims NOT active in sample but active in target:
    # Their contribution is: min(0, t)=0 to intersection, max(0, t)=t to union
    # So we need to add sum(target) over non-sample dims to union.
    # target_sum_total - sum_of_target_at_sample_indices
    target_at_sample = sum(float(target_vector[i]) for i in indices)
    target_total = float(target_vector.sum())
    remaining_target_union = target_total - target_at_sample

    total_union = sample_sum_max + remaining_target_union
    total_intersection = sample_sum_min  # remaining dims contribute 0

    if total_union <= 0:
        return 0.0
    return float(total_intersection / total_union)


def compute_mona_score_arrays(
    samples: Sequence[CanonicalSample],
    target_vectors: Dict[str, np.ndarray],
    *,
    chunk_size: int = 4096,
) -> Dict[str, np.ndarray]:
    """Compute MONA scores for many samples against many target vectors.

    This batches the Jaccard calculation into NumPy array operations so the
    MCTS path does not need one Python loop per benchmark over the full train
    pool. The sample sparse representation remains compact; only chunk-local
    padded arrays are materialized.
    """
    if not target_vectors:
        return {}

    names = list(target_vectors.keys())
    target_matrix = np.stack(
        [np.asarray(target_vectors[name], dtype=np.float32) for name in names],
        axis=0,
    )
    target_totals = target_matrix.sum(axis=1, dtype=np.float64)
    results = {
        name: np.zeros(len(samples), dtype=np.float32)
        for name in names
    }
    if not samples:
        return results

    for start in range(0, len(samples), chunk_size):
        end = min(len(samples), start + chunk_size)
        chunk = samples[start:end]
        max_k = 0
        sparse_chunk: List[Optional[Dict[str, list]]] = []
        for sample in chunk:
            topk = get_sparse_features(sample)
            sparse_chunk.append(topk)
            if topk and topk.get("indices"):
                max_k = max(max_k, min(len(topk["indices"]), len(topk.get("values", []))))

        if max_k <= 0:
            continue

        indices = np.zeros((len(chunk), max_k), dtype=np.int32)
        values = np.zeros((len(chunk), max_k), dtype=np.float32)
        mask = np.zeros((len(chunk), max_k), dtype=bool)

        for row, topk in enumerate(sparse_chunk):
            if not topk:
                continue
            idx_list = topk.get("indices") or []
            val_list = topk.get("values") or []
            k = min(len(idx_list), len(val_list), max_k)
            if k <= 0:
                continue
            indices[row, :k] = np.asarray(idx_list[:k], dtype=np.int32)
            values[row, :k] = np.asarray(val_list[:k], dtype=np.float32)
            mask[row, :k] = True

        if not mask.any():
            continue

        target_at_indices = np.take(target_matrix, indices, axis=1)
        values_3d = values[None, :, :]
        mask_3d = mask[None, :, :]

        intersections = np.minimum(values_3d, target_at_indices) * mask_3d
        max_terms = np.maximum(values_3d, target_at_indices) * mask_3d

        target_at_sample = (target_at_indices * mask_3d).sum(axis=2, dtype=np.float64)
        unions = max_terms.sum(axis=2, dtype=np.float64) + (target_totals[:, None] - target_at_sample)
        numerators = intersections.sum(axis=2, dtype=np.float64)

        chunk_scores = np.divide(
            numerators,
            unions,
            out=np.zeros_like(numerators, dtype=np.float64),
            where=unions > 0,
        )

        for idx, name in enumerate(names):
            results[name][start:end] = chunk_scores[idx].astype(np.float32, copy=False)

    return results


def compute_mona_scores(
    samples: Sequence[CanonicalSample],
    target_vector: np.ndarray,
) -> Tuple[float, float]:
    """Compute MONA score_mean/std for a dataset using cached sparse features.

    Args:
        samples: Dataset with cached sae_topk sparse features.
        target_vector: Dense (d_sae,) eval target vector from mean-pooling
                       eval set SAE features.

    Returns:
        (score_mean, score_std) — aggregated per-sample Jaccard scores.
    """
    all_scores = compute_mona_score_arrays(samples, {"combined": target_vector})["combined"]
    scores = all_scores[all_scores > 0]

    if scores.size == 0:
        return 0.0, 0.0

    mean = float(scores.mean())
    if scores.size == 1:
        return mean, 0.0
    variance = float(scores.var())
    return mean, math.sqrt(max(variance, 0.0))



# -----------------------------------------------------------------------
#  Batch Ingest: Extract & Cache Sparse Features for All Samples
# -----------------------------------------------------------------------

def _detect_idle_devices(
    device: Optional[str] = None,
    min_free_mb: float = 8000.0,
    max_util_pct: float = 30.0,
    max_workers: Optional[int] = None,
) -> List[str]:
    """Auto-detect idle GPU/NPU devices using the unified device selector.

    Returns a list of device strings like ["cuda:0", "cuda:2", "cuda:5"].
    Falls back to ["cpu"] if no accelerators found.
    """
    # If user explicitly specified a device, use it
    if device and device not in {"auto", "cuda", "npu"}:
        return [device]

    try:
        from recipe_sandbox.evaluation.npu_selector import select_idle_devices
        devices = select_idle_devices(
            min_free_mb=min_free_mb,
            max_util_pct=max_util_pct,
            max_workers=max_workers,
        )
        if devices:
            return devices
    except Exception as e:
        logger.warning("select_idle_devices failed: %s. Trying manual detection.", e)

    # Fallback: detect via torch
    import torch
    if torch.cuda.is_available():
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]

    npu = getattr(torch, "npu", None)
    if npu and hasattr(npu, "is_available") and npu.is_available():
        count = int(npu.device_count()) if hasattr(npu, "device_count") else 1
        return [f"npu:{i}" for i in range(count)]

    return ["cpu"]


def _shard_samples(
    samples: List[CanonicalSample],
    n_shards: int,
) -> List[List[int]]:
    """Split sample indices into roughly equal shards."""
    indices = list(range(len(samples)))
    shard_size = max(1, len(indices) // n_shards)
    shards = []
    for i in range(n_shards):
        start = i * shard_size
        end = start + shard_size if i < n_shards - 1 else len(indices)
        if start < len(indices):
            shards.append(indices[start:end])
    return shards


def _ingest_worker(payload: dict) -> List[Dict[str, Any]]:
    """Worker: extract SAE sparse features on one device WITHOUT densifying.

    Key optimization: SAE outputs sparse (top_acts, top_indices) with K=192.
    Instead of densifying to (B, seq_len, d_sae) on GPU (→ OOM),
    we aggregate the sparse outputs on CPU using numpy scatter.
    Memory: ~12MB/batch instead of ~8.6GB/batch.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset
    from recipe_sandbox.scoring.mona import MonaFeatureExtractor

    device_str = payload["device"]
    texts = payload["texts"]
    response_texts = payload.get("response_texts", [])  # For IFD
    compute_ifd = payload.get("compute_ifd", False) and len(response_texts) == len(texts)
    indices = payload["indices"]
    top_k = payload["top_k"]
    batch_size = payload["batch_size"]
    d_sae = payload["d_sae"]
    max_length = payload.get("max_length", 2048)

    print(f"[SAE ingest][{device_str}] Starting: {len(texts)} samples, d_sae={d_sae}, ifd={compute_ifd}", flush=True)

    # Load model + SAE
    extractor = MonaFeatureExtractor.from_paths(
        model_path=payload["model_path"],
        sae_path=payload["sae_path"],
        d_sae=d_sae,
        device=device_str,
        max_length=payload.get("max_length", 2048),
        hidden_state_index=payload.get("hidden_state_index", -2),
        torch_dtype=payload.get("torch_dtype", "bfloat16"),
        hf_home=payload.get("hf_home"),
        device_map=None,
    )
    # Build DataLoader
    class _Texts(Dataset):
        def __init__(self, t): self._t = t
        def __len__(self): return len(self._t)
        def __getitem__(self, i): 
            text = self._t[i]
            return text if text.strip() else " "

    loader = DataLoader(
        _Texts(texts),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: extractor.tokenizer(
            batch, return_tensors="pt", truncation=True,
            max_length=extractor.max_length, padding=True,
        ),
    )

    all_sparse: List[Dict[str, list]] = []
    all_ifd: List[Optional[float]] = []  # IFD scores per sample
    all_ve: List[Optional[float]] = []   # Varentropy scores per sample

    with torch.inference_mode():
        for batch_idx, batch_inputs in enumerate(loader):
            # Move to device and ensure correct dtypes
            for k, v in batch_inputs.items():
                if not hasattr(v, "to"):
                    continue
                if k in ["input_ids", "attention_mask", "position_ids"]:
                    batch_inputs[k] = v.to(device=torch.device(device_str), dtype=torch.long)
                else:
                    batch_inputs[k] = v.to(device_str)

            # Model forward → hidden states
            outputs = extractor.model(**batch_inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[extractor.hidden_state_index]

            # SAE encode → sparse (top_acts, top_indices)
            encoded = extractor.sae.encode(hidden_states)

            if torch.is_tensor(encoded):
                # Dense output → mean over seq → sparse on CPU
                dense_mean = encoded.mean(dim=1).to(torch.float32).cpu().numpy()
                for i in range(dense_mean.shape[0]):
                    nonzero_mask = dense_mean[i] > 0
                    all_sparse.append({
                        "indices": np.where(nonzero_mask)[0].tolist(),
                        "values": dense_mean[i][nonzero_mask].tolist()
                    })
            else:
                # Sparse output → aggregate on CPU without densification!
                if hasattr(encoded, "top_acts"):
                    top_acts = encoded.top_acts.to(torch.float32).cpu().numpy()
                    top_indices_t = encoded.top_indices.cpu().numpy()
                else:
                    top_acts = encoded[0].to(torch.float32).cpu().numpy()
                    top_indices_t = encoded[1].cpu().numpy()

                # top_acts:    (B, seq_len, K)
                # top_indices: (B, seq_len, K)
                B = top_acts.shape[0]
                for i in range(B):
                    # Aggregate across sequence positions on CPU
                    acts_i = top_acts[i]      # (seq_len, K)
                    idxs_i = top_indices_t[i]  # (seq_len, K)
                    seq_len = acts_i.shape[0]

                    # Accumulate: for each feature dim, sum activations
                    acc = np.zeros(d_sae, dtype=np.float64)
                    
                    flat_idxs = idxs_i.flatten()
                    flat_acts = acts_i.flatten()

                    np.add.at(acc, flat_idxs, flat_acts)

                    # Mean: divide by seq_len (not count, to match dense mean)
                    mean_feat = (acc / seq_len).astype(np.float32)

                    nonzero_mask = mean_feat > 0
                    all_sparse.append({
                        "indices": np.where(nonzero_mask)[0].tolist(),
                        "values": mean_feat[nonzero_mask].tolist()
                    })

            # Log progress
            done = len(all_sparse)

            # --- IFD piggybacking ---
            if compute_ifd:
                batch_start = done - (top_acts.shape[0] if not torch.is_tensor(encoded) else dense_mean.shape[0])
                batch_end = done
                batch_full = texts[batch_start:batch_end]
                batch_resp = response_texts[batch_start:batch_end]
                valid_mask = [bool(r.strip()) for r in batch_resp]
                if any(valid_mask):
                    ifd_scores, ve_scores = _compute_ifd_for_batch(
                        extractor.model, extractor.tokenizer,
                        batch_full, batch_resp, device_str, max_length,
                    )
                    for i, (score, ve) in enumerate(zip(ifd_scores, ve_scores)):
                        if valid_mask[i]:
                            all_ifd.append(round(score, 6))
                            all_ve.append(round(ve, 6))
                        else:
                            all_ifd.append(None)
                            all_ve.append(None)
                else:
                    all_ifd.extend([None] * (batch_end - batch_start))
                    all_ve.extend([None] * (batch_end - batch_start))
            else:
                batch_count = top_acts.shape[0] if not torch.is_tensor(encoded) else dense_mean.shape[0]
                all_ifd.extend([None] * batch_count)
                all_ve.extend([None] * batch_count)

            if (batch_idx + 1) % 50 == 0 or done >= len(texts):
                print(f"[SAE ingest][{device_str}] {done}/{len(texts)} samples", flush=True)

    try:
        extractor.close()
    except Exception as e:
        # Cleanup errors should not discard successfully computed results.
        # Just free what we can without crashing the worker.
        print(f"[SAE ingest][{device_str}] WARNING: close() failed (non-fatal): {e}", flush=True)
        try:
            import gc
            extractor.model = None
            extractor.sae = None
            extractor.tokenizer = None
            gc.collect()
        except Exception:
            pass

    # Pack results
    results = []
    for i, idx in enumerate(indices):
        item = {"index": idx, "sparse": all_sparse[i]}
        if i < len(all_ifd) and all_ifd[i] is not None:
            item["ifd_score"] = all_ifd[i]
            item["varentropy_score"] = all_ve[i]
        results.append(item)

    print(f"[SAE ingest][{device_str}] Done: {len(results)} samples", flush=True)
    return results


def ingest_sparse_features(
    samples: List[CanonicalSample],
    *,
    model_path: str,
    sae_path: str,
    d_sae: Optional[int] = None,
    top_k: int = DEFAULT_TOP_K,
    batch_size: int = 8,
    device: Optional[str] = None,
    device_map: Optional[str] = None,
    max_length: int = 2048,
    hidden_state_index: int = -2,
    torch_dtype: str = "bfloat16",
    hf_home: Optional[str] = None,
    show_progress: bool = True,
    max_workers: Optional[int] = None,
    compute_ifd: bool = True,
    compute_cpu_heuristics: bool = True,
    cpu_max_workers: Optional[int] = None,
) -> int:
    """Run SAE feature extraction and cache sparse top-K in all samples.

    **Multi-GPU/NPU support**: Auto-detects idle devices via
    ``select_idle_devices()``, shards samples evenly across workers,
    and runs parallel extraction with one process per device.

    This is meant to be called ONCE during the Ingest phase.
    After this, all samples have ``metadata.extra["sae_topk"]`` populated.
    If compute_ifd=True, ``metadata.extra["ifd"]["score"]`` is also set.

    Returns:
        Number of samples successfully annotated.
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from recipe_sandbox.operators.helpers import sample_to_text

    # 0. Auto-detect d_sae from SAE config if not provided
    if d_sae is None:
        d_sae = _detect_d_sae(sae_path)
        logger.info("Auto-detected d_sae=%d from SAE config", d_sae)

    # 1. Detect available devices (need ~20GB for model+SAE+overhead)
    devices = _detect_idle_devices(
        device=device,
        min_free_mb=20000.0,
        max_util_pct=30.0,
        max_workers=max_workers,
    )
    n_workers = len(devices)

    logger.info(
        "SAE sparse feature ingest: %d samples, top_k=%d, d_sae=%d, "
        "devices=%s (%d worker(s))",
        len(samples), top_k, d_sae, devices, n_workers,
    )

    # 2. If single device or device_map set, run in-process (simpler)
    if n_workers <= 1 or device_map is not None:
        result = _ingest_single_device(
            samples,
            model_path=model_path,
            sae_path=sae_path,
            d_sae=d_sae,
            top_k=top_k,
            batch_size=batch_size,
            device=devices[0] if devices else "cpu",
            device_map=device_map,
            max_length=max_length,
            hidden_state_index=hidden_state_index,
            torch_dtype=torch_dtype,
            hf_home=hf_home,
            show_progress=show_progress,
            compute_ifd=compute_ifd,
        )
        # Release GPU memory before returning
        try:
            from recipe_sandbox.evaluation.gpu_cleanup import release_all_gpu_memory
            release_all_gpu_memory(wait_seconds=2.0)
        except Exception as exc:
            logger.warning("GPU cleanup failed (non-critical): %s", exc)
        if compute_cpu_heuristics:
            annotate_cpu_heuristics(samples, cpu_max_workers)
        return result

    # 3. Multi-device: shard and spawn workers
    texts = [sample_to_text(s) for s in samples]
    response_texts = [_extract_response_text(s) for s in samples] if compute_ifd else []
    shards = _shard_samples(samples, n_workers)

    logger.info(
        "Sharding %d samples across %d devices: %s",
        len(samples), n_workers,
        ", ".join(f"{d}({len(s)})" for d, s in zip(devices, shards)),
    )

    payloads = [
        {
            "device": dev,
            "texts": [texts[i] for i in shard_indices],
            "indices": shard_indices,
            "top_k": top_k,
            "batch_size": batch_size,
            "model_path": model_path,
            "sae_path": sae_path,
            "d_sae": d_sae,
            "max_length": max_length,
            "hidden_state_index": hidden_state_index,
            "torch_dtype": torch_dtype,
            "hf_home": hf_home,
            "compute_ifd": compute_ifd,
            "response_texts": [response_texts[i] for i in shard_indices] if compute_ifd else [],
        }
        for dev, shard_indices in zip(devices, shards)
        if shard_indices
    ]

    annotated = 0
    with ProcessPoolExecutor(
        max_workers=len(payloads),
        mp_context=mp.get_context("spawn"),
    ) as executor:
        futures = [
            executor.submit(_ingest_worker, payload)
            for payload in payloads
        ]
        for future in futures:
            results = future.result()
            for item in results:
                idx = item["index"]
                samples[idx].metadata.extra[SPARSE_KEY] = item["sparse"]
                # Write back IFD and Varentropy score if present
                if "ifd_score" in item:
                    if "ifd" not in samples[idx].metadata.extra:
                        samples[idx].metadata.extra["ifd"] = {}
                    samples[idx].metadata.extra["ifd"]["score"] = item["ifd_score"]
                if "varentropy_score" in item:
                    if "varentropy" not in samples[idx].metadata.extra:
                        samples[idx].metadata.extra["varentropy"] = {}
                    samples[idx].metadata.extra["varentropy"]["score"] = item["varentropy_score"]
                annotated += 1

    logger.info(
        "Multi-device SAE ingest complete: %d/%d samples on %d device(s)",
        annotated, len(samples), n_workers,
    )

    # Release GPU memory after multi-device ingest
    try:
        from recipe_sandbox.evaluation.gpu_cleanup import release_all_gpu_memory
        release_all_gpu_memory(wait_seconds=2.0)
    except Exception as exc:
        logger.warning("GPU cleanup failed (non-critical): %s", exc)

    annotate_cpu_heuristics(samples, cpu_max_workers)
    
    return annotated


def _ingest_single_device(
    samples: List[CanonicalSample],
    *,
    model_path: str,
    sae_path: str,
    d_sae: Optional[int] = None,
    top_k: int = DEFAULT_TOP_K,
    batch_size: int = 8,
    device: str = "cpu",
    device_map: Optional[str] = None,
    max_length: int = 2048,
    hidden_state_index: int = -2,
    torch_dtype: str = "bfloat16",
    hf_home: Optional[str] = None,
    show_progress: bool = True,
    compute_ifd: bool = True,
) -> int:
    """Single-device SAE ingest — sparse-native, no GPU densification.

    When compute_ifd=True, also computes IFD scores piggybacking on the
    already-loaded model, storing results in metadata.extra.ifd.score.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset
    from recipe_sandbox.scoring.mona import MonaFeatureExtractor
    from recipe_sandbox.operators.helpers import sample_to_text

    logger.info("Single-device SAE ingest on %s (d_sae=%s, ifd=%s)", device, d_sae, compute_ifd)

    extractor = MonaFeatureExtractor.from_paths(
        model_path=model_path,
        sae_path=sae_path,
        d_sae=d_sae,
        device=device,
        max_length=max_length,
        hidden_state_index=hidden_state_index,
        torch_dtype=torch_dtype,
        hf_home=hf_home,
        device_map=device_map,
    )
    resolved_d_sae = d_sae if d_sae is not None else extractor.d_sae
    if resolved_d_sae is None:
        raise ValueError(f"Failed to resolve d_sae for SAE ingest from {sae_path}")
    
    texts = [sample_to_text(s) for s in samples]

    # Pre-extract response texts for IFD (cheap CPU operation)
    response_texts = []
    if compute_ifd:
        response_texts = [_extract_response_text(s) for s in samples]

    class _Texts(Dataset):
        def __init__(self, t): self._t = t
        def __len__(self): return len(self._t)
        def __getitem__(self, i):
            text = self._t[i]
            return text if text.strip() else " "

    loader = DataLoader(
        _Texts(texts),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: extractor.tokenizer(
            batch, return_tensors="pt", truncation=True,
            max_length=extractor.max_length, padding=True,
        ),
    )

    annotated = 0
    ifd_computed = 0
    with torch.inference_mode():
        for batch_idx, batch_inputs in enumerate(loader):
            for k, v in batch_inputs.items():
                if not hasattr(v, "to"):
                    continue
                if k in ["input_ids", "attention_mask", "position_ids"]:
                    batch_inputs[k] = v.to(device=torch.device(device), dtype=torch.long)
                else:
                    batch_inputs[k] = v.to(device)

            outputs = extractor.model(**batch_inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[extractor.hidden_state_index]
            encoded = extractor.sae.encode(hidden_states)

            if torch.is_tensor(encoded):
                dense_mean = encoded.mean(dim=1).to(torch.float32).cpu().numpy()
                for i in range(dense_mean.shape[0]):
                    sample_idx = annotated + i
                    if sample_idx < len(samples):
                        annotate_sample_sparse(samples[sample_idx], dense_mean[i], top_k=top_k)
            else:
                if hasattr(encoded, "top_acts"):
                    top_acts = encoded.top_acts.to(torch.float32).cpu().numpy()
                    top_indices_t = encoded.top_indices.cpu().numpy()
                else:
                    top_acts = encoded[0].to(torch.float32).cpu().numpy()
                    top_indices_t = encoded[1].cpu().numpy()

                B = top_acts.shape[0]
                for i in range(B):
                    sample_idx = annotated + i
                    if sample_idx >= len(samples):
                        break
                    acts_i = top_acts[i]
                    idxs_i = top_indices_t[i]
                    seq_len = acts_i.shape[0]
                    flat_idxs = idxs_i.flatten()
                    flat_acts = acts_i.flatten()
                    
                    # 使用原本的明确累加方式，确保重复 token ID 被相加
                    acc = np.zeros(resolved_d_sae, dtype=np.float64)
                    np.add.at(acc, flat_idxs, flat_acts)
                    
                    # 平均到整个 seq_len
                    mean_feat = (acc / seq_len).astype(np.float32)
                    
                    # 核心：保留所有的非零特征，避免 top_k 强制腰斩
                    nonzero_mask = mean_feat > 0
                    samples[sample_idx].metadata.extra["sae_topk"] = {
                        "indices": np.where(nonzero_mask)[0].tolist(),
                        "values": mean_feat[nonzero_mask].tolist()
                    }

            batch_count = top_acts.shape[0] if not torch.is_tensor(encoded) else dense_mean.shape[0]
            # --- IFD piggybacking: compute while model is hot in GPU ---
            if compute_ifd:
                batch_start = annotated
                batch_end = min(annotated + batch_count, len(samples))
                batch_full = texts[batch_start:batch_end]
                batch_resp = response_texts[batch_start:batch_end]
                # Only compute for samples that have non-empty responses
                valid_mask = [bool(r.strip()) for r in batch_resp]
                if any(valid_mask):
                    ifd_scores, ve_scores = _compute_ifd_for_batch(
                        extractor.model, extractor.tokenizer,
                        batch_full, batch_resp, device, max_length,
                    )
                    for i, (score, ve) in enumerate(zip(ifd_scores, ve_scores)):
                        si = batch_start + i
                        if si < len(samples) and valid_mask[i]:
                            if "ifd" not in samples[si].metadata.extra:
                                samples[si].metadata.extra["ifd"] = {}
                            if "varentropy" not in samples[si].metadata.extra:
                                samples[si].metadata.extra["varentropy"] = {}
                            samples[si].metadata.extra["ifd"]["score"] = round(score, 6)
                            samples[si].metadata.extra["varentropy"]["score"] = round(ve, 6)
                            ifd_computed += 1

            annotated += batch_count

            if show_progress and ((batch_idx + 1) % 50 == 0 or annotated >= len(samples)):
                logger.info("[%s] %d/%d samples (ifd: %d)", device, annotated, len(samples), ifd_computed)

    extractor.close()

    logger.info(
        "Sparse feature ingest complete: %d/%d samples (top_k=%d, ifd=%d)",
        annotated, len(samples), top_k, ifd_computed,
    )
    return annotated


# -----------------------------------------------------------------------
#  Precomputed Reference Stats (for search loop)
# -----------------------------------------------------------------------

class SparseFeatureCache:
    """Precomputed reference statistics for fast drift/entropy during search.

    Usage:
        # During ingest (once):
        cache = SparseFeatureCache.from_samples(all_samples, d_sae=detected_d_sae,
                                                 eval_target_vector=eval_target)

        # During search loop (milliseconds):
        drift = cache.compute_drift(filtered_samples)
        entropy = cache.compute_entropy(filtered_samples)
        score_mean, score_std = cache.compute_scores(filtered_samples)
    """

    def __init__(
        self,
        reference_snar: np.ndarray,
        reference_mean_act: np.ndarray,
        reference_entropy: float,
        d_sae: int,
        n_samples: int,
        eval_target_vector: Optional[np.ndarray] = None,
        sample_sparse_indices: Optional[list] = None,
        sample_sparse_values: Optional[list] = None,
    ):
        self.reference_snar = reference_snar
        self.reference_mean_act = reference_mean_act
        self.reference_entropy = reference_entropy
        self.d_sae = d_sae
        self.n_samples = n_samples
        self.eval_target_vector = eval_target_vector  # (d_sae,) or None
        self.sample_sparse_indices = sample_sparse_indices
        self.sample_sparse_values = sample_sparse_values

    @classmethod
    def from_samples(
        cls,
        samples: Sequence[CanonicalSample],
        d_sae: int,
        eval_target_vector: Optional[np.ndarray] = None,
    ) -> "SparseFeatureCache":
        """Build reference cache from samples with cached sparse features."""
        snar = compute_snar(samples, d_sae)
        mean_act = compute_mean_activation(samples, d_sae)
        entropy = sparse_activation_entropy(snar)

        sample_sparse_indices = []
        sample_sparse_values = []
        for s in samples:
            topk = s.metadata.extra.get("sae_topk")
            if topk:
                sample_sparse_indices.append(np.array(topk["indices"], dtype=np.int32))
                sample_sparse_values.append(np.array(topk["values"], dtype=np.float32))
            else:
                sample_sparse_indices.append(np.array([], dtype=np.int32))
                sample_sparse_values.append(np.array([], dtype=np.float32))

        logger.info(
            "SparseFeatureCache built: %d samples, d_sae=%d, "
            "active_dims=%d, entropy=%.3f, has_eval_target=%s, has_raw_features=%s",
            len(samples), d_sae,
            int((snar > 0).sum()), entropy,
            eval_target_vector is not None,
            len(sample_sparse_indices) > 0,
        )
        return cls(
            reference_snar=snar,
            reference_mean_act=mean_act,
            reference_entropy=entropy,
            d_sae=d_sae,
            n_samples=len(samples),
            eval_target_vector=eval_target_vector,
            sample_sparse_indices=sample_sparse_indices,
            sample_sparse_values=sample_sparse_values,
        )

    def compute_drift(self, filtered_samples: Sequence[CanonicalSample]) -> float:
        """Compute drift between reference and filtered dataset (milliseconds)."""
        current_snar = compute_snar(filtered_samples, self.d_sae)
        return sparse_distribution_drift(self.reference_snar, current_snar)

    def compute_entropy(self, filtered_samples: Sequence[CanonicalSample]) -> float:
        """Compute SAE activation entropy for filtered dataset."""
        current_snar = compute_snar(filtered_samples, self.d_sae)
        return sparse_activation_entropy(current_snar)

    def compute_scores(self, filtered_samples: Sequence[CanonicalSample]) -> Tuple[float, float]:
        """Compute MONA score_mean/std from cached sparse features.

        Uses generalized Jaccard similarity between each sample's cached
        sae_topk and the eval target vector.

        Returns (0.0, 0.0) if no eval target vector is set.
        """
        if self.eval_target_vector is None:
            return 0.0, 0.0
        return compute_mona_scores(filtered_samples, self.eval_target_vector)

    def compute_jaccard_vs_reference(self, filtered_samples: Sequence[CanonicalSample]) -> float:
        """Generalized Jaccard similarity of filtered vs reference mean activations."""
        current_mean = compute_mean_activation(filtered_samples, self.d_sae)
        return sparse_generalized_jaccard(self.reference_mean_act, current_mean)

    def save(self, path: str) -> None:
        """Save cache to disk."""
        save_dict = dict(
            reference_snar=self.reference_snar,
            reference_mean_act=self.reference_mean_act,
            reference_entropy=np.array([self.reference_entropy]),
            d_sae=np.array([self.d_sae]),
            n_samples=np.array([self.n_samples]),
        )
        if self.eval_target_vector is not None:
            save_dict["eval_target_vector"] = self.eval_target_vector
            
        if self.sample_sparse_indices is not None and len(self.sample_sparse_indices) > 0:
            save_dict["sample_sparse_indices"] = np.array(self.sample_sparse_indices, dtype=object)
            save_dict["sample_sparse_values"] = np.array(self.sample_sparse_values, dtype=object)
            
        np.savez_compressed(path, **save_dict)
        logger.info("SparseFeatureCache saved → %s", path)

    @classmethod
    def load(cls, path: str) -> "SparseFeatureCache":
        """Load cache from disk."""
        data = np.load(path, allow_pickle=True)
        eval_tv = data["eval_target_vector"] if "eval_target_vector" in data else None
        
        sample_sparse_indices = None
        sample_sparse_values = None
        if "sample_sparse_indices" in data and "sample_sparse_values" in data:
            sample_sparse_indices = list(data["sample_sparse_indices"])
            sample_sparse_values = list(data["sample_sparse_values"])
            
        return cls(
            reference_snar=data["reference_snar"],
            reference_mean_act=data["reference_mean_act"],
            reference_entropy=float(data["reference_entropy"][0]),
            d_sae=int(data["d_sae"][0]),
            n_samples=int(data["n_samples"][0]),
            eval_target_vector=eval_tv,
            sample_sparse_indices=sample_sparse_indices,
            sample_sparse_values=sample_sparse_values,
        )

    def hydrate_samples(self, samples: Sequence[CanonicalSample]) -> None:
        """Hydrate samples with cached per-sample sparse features."""
        if not self.sample_sparse_indices or not self.sample_sparse_values:
            return
        hydrated = 0
        for i, sample in enumerate(samples):
            if i >= len(self.sample_sparse_indices):
                break
            idx = self.sample_sparse_indices[i].tolist()
            if not idx:
                continue
            sample.metadata.extra["sae_topk"] = {
                "indices": idx,
                "values": self.sample_sparse_values[i].tolist(),
            }
            hydrated += 1
        if hydrated > 0:
            logger.info("Hydrated %d samples from per-sample *.npz cache features", hydrated)


# -----------------------------------------------------------------------
#  Internal helpers
# -----------------------------------------------------------------------

def _cpu_heuristic_worker(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Worker function for computing CPU heuristics (N-Gram + Action/Object)."""
    import math
    import re
    from collections import Counter
    
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
    except ImportError:
        nlp = None

    texts = payload["texts"]
    indices = payload["indices"]
    
    results = []
    for i, text in enumerate(texts):
        res = {"index": indices[i]}
        
        # 1. N-Gram Entropy
        tokens = re.findall(r'\b\w+\b', text.lower())
        if tokens:
            counts = Counter(tokens)
            total = len(tokens)
            entropy = -sum((c/total) * math.log2(c/total) for c in counts.values())
            res["ngram_entropy"] = entropy
        else:
            res["ngram_entropy"] = 0.0
            
        # 2. Action-Object Branching
        if nlp is not None and text:
            doc = nlp(text)
            verbs = [token for token in doc if token.pos_ == "VERB"]
            if verbs:
                avg_subtree_len = sum(len(list(v.subtree)) for v in verbs) / len(verbs)
                res["action_object"] = len(verbs) * 0.5 + avg_subtree_len * 0.5
            else:
                res["action_object"] = 0.0
        else:
            res["action_object"] = 0.0
            
        results.append(res)
    return results


def annotate_cpu_heuristics(samples: List['CanonicalSample'], max_workers: Optional[int] = None):
    """Run CPU-bound heuristic annotation using multiprocessing."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import os
    
    # We only need the message text for CPU heuristics
    texts = []
    for s in samples:
        texts.append(" ".join([m.content for m in s.messages] + [s.target.text if s.target.text else ""]))
        
    n_workers = max_workers or os.cpu_count() or 4
    n_workers = min(n_workers, len(samples))
    
    if n_workers <= 0 or len(samples) == 0:
        return
        
    chunk_size = max(1, len(samples) // n_workers)
    shards = [list(range(i, min(i + chunk_size, len(samples)))) for i in range(0, len(samples), chunk_size)]
    
    payloads = [
        {"texts": [texts[i] for i in shard], "indices": shard}
        for shard in shards if shard
    ]
    
    logger.info("Computing CPU Heuristics (N-Gram Entropy & ACT Branching) on %d samples using %d workers ...", len(samples), len(payloads))
    
    completed = 0
    total = len(samples)
    with ProcessPoolExecutor(max_workers=len(payloads), mp_context=mp.get_context("spawn")) as executor:
        futures = {executor.submit(_cpu_heuristic_worker, p): len(p["indices"]) for p in payloads}
        for future in as_completed(futures):
            results = future.result()
            batch_size = futures[future]
            for r in results:
                idx = r["index"]
                if "ngram_entropy" not in samples[idx].metadata.extra:
                    samples[idx].metadata.extra["ngram_entropy"] = {}
                samples[idx].metadata.extra["ngram_entropy"]["score"] = r["ngram_entropy"]
                
                if "action_object" not in samples[idx].metadata.extra:
                    samples[idx].metadata.extra["action_object"] = {}
                samples[idx].metadata.extra["action_object"]["score"] = r["action_object"]
            completed += batch_size
            logger.info("  [CPU Heuristics] Progress: %d / %d (%.1f%%)", completed, total, completed / total * 100)

    logger.info("CPU Heuristics computation complete: %d samples annotated.", total)


def _detect_d_sae(sae_path: str) -> int:
    """Detect d_sae dimension from the SAE checkpoint directory.

    Reads ``cfg.json`` in the SAE directory to compute
    ``d_sae = d_in * expansion_factor``.

    Falls back to inspecting safetensors weight shapes, or the
    ``sparsify`` package if available.
    """
    import json as _json
    from pathlib import Path

    sae_dir = Path(sae_path)

    # Method 1: Read cfg.json (most common for EleutherAI SAEs)
    cfg_path = sae_dir / "cfg.json" if sae_dir.is_dir() else sae_dir.parent / "cfg.json"
    if cfg_path.exists():
        cfg = _json.loads(cfg_path.read_text())
        d_in = cfg.get("d_in")
        expansion_factor = cfg.get("expansion_factor")
        if d_in and expansion_factor:
            d_sae = int(d_in) * int(expansion_factor)
            logger.info("d_sae=%d (d_in=%s × expansion_factor=%s) from %s", d_sae, d_in, expansion_factor, cfg_path)
            return d_sae
        # Some configs store d_sae directly
        if "d_sae" in cfg:
            return int(cfg["d_sae"])

    # Method 2: Read safetensors metadata / weight shapes
    safetensors_path = sae_dir / "sae.safetensors" if sae_dir.is_dir() else sae_dir
    if safetensors_path.exists() and safetensors_path.suffix == ".safetensors":
        try:
            from safetensors import safe_open
            with safe_open(str(safetensors_path), framework="pt") as f:
                # W_dec typically has shape (d_sae, d_in) or (d_in, d_sae)
                for key in ["W_dec", "decoder.weight", "W_enc"]:
                    if key in f.keys():
                        shape = f.get_slice(key).get_shape()
                        d_sae = max(shape)  # d_sae is the larger dimension
                        logger.info("d_sae=%d from safetensors key '%s' shape %s", d_sae, key, shape)
                        return d_sae
        except Exception as e:
            logger.warning("Could not read safetensors for d_sae: %s", e)

    # Method 3: Try sparsify package
    try:
        from sparsify import Sae
        sae = Sae.load_from_disk(sae_path)
        if hasattr(sae, "cfg") and hasattr(sae.cfg, "d_sae"):
            d_sae = int(sae.cfg.d_sae)
        elif hasattr(sae, "d_sae"):
            d_sae = int(sae.d_sae)
        elif hasattr(sae, "W_dec"):
            d_sae = max(sae.W_dec.shape)
        else:
            raise ValueError("Cannot detect d_sae from sparsify SAE")
        del sae
        import gc; gc.collect()
        return d_sae
    except ImportError:
        pass

    raise ValueError(
        f"Cannot auto-detect d_sae from {sae_path}. "
        "Pass d_sae explicitly via --d_sae or TaskConfig.model.d_sae."
    )

def _to_numpy(x: Any) -> np.ndarray:
    """Convert tensor/list to numpy array."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):  # torch.Tensor
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)
