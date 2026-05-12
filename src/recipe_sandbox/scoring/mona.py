from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import gc
import hashlib
import multiprocessing as mp
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

from recipe_sandbox.operators.helpers import sample_to_text
from recipe_sandbox.schema.types import CanonicalSample


DEFAULT_TARGET_VECTOR_MAX_SAMPLES = 128


class MonaFeatureExtractor:
    def __init__(
        self,
        model,
        tokenizer,
        sae,
        *,
        d_sae: Optional[int] = None,
        device: Optional[str] = None,
        max_length: int = 2048,
        hidden_state_index: int = -2,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.sae = sae
        self.d_sae = d_sae
        self.device = device or self._infer_device(model)
        self.max_length = max_length
        self.hidden_state_index = hidden_state_index

    def extract_sample_feature(self, sample: CanonicalSample) -> torch.Tensor:
        return self.extract_text_feature(sample_to_text(sample))

    def close(self) -> None:
        device = getattr(self, "device", None)
        model = getattr(self, "model", None)
        sae = getattr(self, "sae", None)
        tokenizer = getattr(self, "tokenizer", None)

        self.model = None
        self.sae = None
        self.tokenizer = None

        del model
        del sae
        del tokenizer
        _release_torch_resources(device=device)

    def extract_dataset_features(self, dataset: Sequence[CanonicalSample]) -> List[torch.Tensor]:
        return [self.extract_sample_feature(sample) for sample in dataset]

    def extract_batch_features(
        self,
        texts: List[str],
        batch_size: int = 8,
        *,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
    ) -> torch.Tensor:
        """Batch feature extraction using DataLoader for efficiency.

        Returns (N, D_SAE) tensor of SAE features for all texts.
        """
        dataset = _TextListDataset(texts)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda batch: self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            ),
        )

        all_embeddings: List[torch.Tensor] = []
        with torch.inference_mode():
            for batch_inputs in _progress_iterable(
                loader,
                enabled=show_progress,
                total=len(loader),
                desc=progress_desc or f"Extracting features on {self.device}",
                unit="batch",
            ):
                batch_inputs = self._move_to_device(batch_inputs, self.device)
                outputs = self.model(**batch_inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[self.hidden_state_index]
                encoded = self.sae.encode(hidden_states)
                dense = self._to_dense_activations(encoded, hidden_states.device)
                features = dense.mean(dim=1).to(torch.float32)  # (B, D_SAE)
                all_embeddings.append(features.cpu())

        return torch.cat(all_embeddings, dim=0)

    def build_target_vector(
        self,
        dataset: Sequence[CanonicalSample],
        *,
        batch_size: int = 8,
        max_samples: Optional[int] = DEFAULT_TARGET_VECTOR_MAX_SAMPLES,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
        sample_seed: int = 42,
        sample_label: str = "target",
    ) -> torch.Tensor:
        """Build a single target vector as the mean of sample features.

        Uses batched extraction when *batch_size* > 1.
        """
        samples = list(dataset)
        if not samples:
            raise ValueError("target dataset must contain at least one sample")
        samples = _sample_canonical_samples(
            samples,
            max_samples=max_samples,
            sample_seed=sample_seed,
            sample_label=sample_label,
        )

        texts = [sample_to_text(s) for s in samples]
        if batch_size > 1:
            embeddings = self.extract_batch_features(
                texts,
                batch_size=batch_size,
                show_progress=show_progress,
                progress_desc=progress_desc,
            )
            return embeddings.mean(dim=0)

        features = []
        iterator = _progress_iterable(
            texts,
            enabled=show_progress,
            total=len(texts),
            desc=progress_desc or f"Extracting features on {self.device}",
            unit="sample",
        )
        for text in iterator:
            features.append(self.extract_text_feature(text))
        return torch.stack(features, dim=0).mean(dim=0)

    def build_target_vectors(
        self,
        datasets: Dict[str, Sequence[CanonicalSample]],
        *,
        batch_size: int = 8,
        max_samples: Optional[int] = DEFAULT_TARGET_VECTOR_MAX_SAMPLES,
        show_progress: bool = False,
        sample_seed: int = 42,
    ) -> Dict[str, torch.Tensor]:
        """Build target vectors for multiple eval tasks.

        Parameters
        ----------
        datasets : dict[str, Sequence[CanonicalSample]]
            Mapping of task_name → eval samples.
        batch_size : int
            Batch size for feature extraction.
        max_samples : int, optional
            Max samples per task.

        Returns
        -------
        dict[str, Tensor]
            task_name → target vector (D_SAE,)
        """
        target_vectors: Dict[str, torch.Tensor] = {}
        total_tasks = len(datasets)
        for task_index, (task_name, samples) in enumerate(datasets.items(), start=1):
            if show_progress:
                sample_count = len(samples) if hasattr(samples, "__len__") else "?"
                print(
                    f"[MONA] Building target vector {task_index}/{total_tasks} "
                    f"for {task_name} ({sample_count} eval samples, device={self.device})",
                    flush=True,
                )
            target_vectors[task_name] = self.build_target_vector(
                samples,
                batch_size=batch_size,
                max_samples=max_samples,
                show_progress=show_progress,
                progress_desc=f"Building target vector for {task_name} on {self.device}",
                sample_seed=sample_seed,
                sample_label=task_name,
            )
            if show_progress:
                print(
                    f"[MONA] Finished target vector {task_index}/{total_tasks} for {task_name}",
                    flush=True,
                )
        return target_vectors

    @classmethod
    def build_target_vectors_multi_device(
        cls,
        *,
        model_path: str,
        sae_path: str,
        datasets: Dict[str, Sequence[CanonicalSample]],
        devices: Sequence[str],
        d_sae: Optional[int] = None,
        max_length: int = 2048,
        hidden_state_index: int = -2,
        torch_dtype: str = "bfloat16",
        hf_home: Optional[str] = None,
        device_map: Optional[str] = None,
        batch_size: int = 8,
        max_samples: Optional[int] = DEFAULT_TARGET_VECTOR_MAX_SAMPLES,
        show_progress: bool = True,
        sample_seed: int = 42,
    ) -> Dict[str, torch.Tensor]:
        """Build target vectors across multiple devices.

        Each device gets a subset of tasks and loads one dedicated extractor.
        This is intended for many eval tasks on multi-GPU hosts.
        """
        if not datasets:
            return {}

        normalized_devices = [str(device) for device in devices if str(device)]
        if len(normalized_devices) <= 1 or len(datasets) <= 1:
            extractor = cls.from_paths(
                model_path=model_path,
                sae_path=sae_path,
                d_sae=d_sae,
                device=normalized_devices[0] if normalized_devices else None,
                max_length=max_length,
                hidden_state_index=hidden_state_index,
                torch_dtype=torch_dtype,
                hf_home=hf_home,
                device_map=device_map,
            )
            return extractor.build_target_vectors(
                datasets,
                batch_size=batch_size,
                max_samples=max_samples,
                show_progress=show_progress,
                sample_seed=sample_seed,
            )

        assignments = _partition_named_datasets(datasets, normalized_devices)
        target_vectors: Dict[str, torch.Tensor] = {}
        with ProcessPoolExecutor(
            max_workers=len(assignments),
            mp_context=mp.get_context("spawn"),
        ) as executor:
            futures = [
                executor.submit(
                    _build_target_vector_worker,
                    {
                        "model_path": model_path,
                        "sae_path": sae_path,
                        "dataset_items": shard,
                        "device": device,
                        "d_sae": d_sae,
                        "max_length": max_length,
                        "hidden_state_index": hidden_state_index,
                        "torch_dtype": torch_dtype,
                        "hf_home": hf_home,
                        "batch_size": batch_size,
                        "max_samples": max_samples,
                        "show_progress": show_progress,
                        "sample_seed": sample_seed,
                    },
                )
                for device, shard in assignments
                if shard
            ]
            for future in futures:
                target_vectors.update(future.result())
        return target_vectors

    def extract_text_feature(self, text: str) -> torch.Tensor:
        batch = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        batch = self._move_to_device(batch, self.device)

        with torch.inference_mode():
            outputs = self.model(**batch, output_hidden_states=True)
            hidden_states = outputs.hidden_states[self.hidden_state_index]
            encoded = self.sae.encode(hidden_states)
            dense = self._to_dense_activations(encoded, hidden_states.device)
            feature = dense.mean(dim=1).squeeze(0).to(torch.float32)
        return feature

    def _to_dense_activations(self, encoded, device: torch.device) -> torch.Tensor:
        if torch.is_tensor(encoded):
            return encoded.to(torch.float32)

        if isinstance(encoded, tuple) or hasattr(encoded, "top_acts"):
            top_acts = encoded.top_acts if hasattr(encoded, "top_acts") else encoded[0]
            top_indices = encoded.top_indices if hasattr(encoded, "top_indices") else encoded[1]
            d_sae = self.d_sae
            if d_sae is None:
                if hasattr(self.sae, "cfg") and hasattr(self.sae.cfg, "d_sae"):
                    d_sae = int(self.sae.cfg.d_sae)
                elif hasattr(self.sae, "d_sae"):
                    d_sae = int(self.sae.d_sae)
                else:
                    raise ValueError("d_sae is required when SAE encode output is sparse")

            batch_size, seq_len, _ = top_acts.shape
            dense = torch.zeros((batch_size, seq_len, d_sae), dtype=torch.float32, device=device)
            dense.scatter_(dim=-1, index=top_indices, src=top_acts.to(torch.float32))
            return dense

        raise TypeError("Unsupported SAE encode output type")

    def _infer_device(self, model) -> str:
        try:
            parameter = next(model.parameters())
            return str(parameter.device)
        except (AttributeError, StopIteration, TypeError):
            return "cpu"

    def _move_to_device(self, batch, device: str):
        if hasattr(batch, "to"):
            return batch.to(device)
        if isinstance(batch, dict):
            return {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }
        return batch

    @classmethod
    def from_paths(
        cls,
        *,
        model_path: str,
        sae_path: str,
        d_sae: Optional[int] = None,
        device: Optional[str] = None,
        max_length: int = 2048,
        hidden_state_index: int = -2,
        torch_dtype: str = "bfloat16",
        hf_home: Optional[str] = None,
        device_map: Optional[str] = None,
    ) -> "MonaFeatureExtractor":
        if hf_home:
            os.environ["HF_HOME"] = hf_home

        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            from sparsify import Sae
        except ImportError as exc:
            raise ImportError("MonaFeatureExtractor.from_paths requires the sparsify package") from exc

        dtype = _resolve_torch_dtype(torch_dtype)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {"torch_dtype": dtype}
        if device_map is not None:
            model_kwargs["device_map"] = device_map
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        resolved_device = device or _infer_runtime_device(model)
        if device_map is None and hasattr(model, "to"):
            model = model.to(resolved_device)
        model.eval()

        sae = Sae.load_from_disk(sae_path)
        if hasattr(sae, "to"):
            sae = sae.to(resolved_device)

        return cls(
            model=model,
            tokenizer=tokenizer,
            sae=sae,
            d_sae=d_sae,
            device=resolved_device,
            max_length=max_length,
            hidden_state_index=hidden_state_index,
        )


@dataclass
class MonaScoredRecord:
    sample_id: str
    similarities: Dict[str, float]
    feature: torch.Tensor


class MonaScorer:
    def __init__(self, extractor: MonaFeatureExtractor, target_vectors: Dict[str, torch.Tensor]) -> None:
        self.extractor = extractor
        self.target_vectors = {
            task_name: vector.to(dtype=torch.float32, device=self._device())
            for task_name, vector in target_vectors.items()
        }

    def close(self) -> None:
        extractor = getattr(self, "extractor", None)
        target_vectors = getattr(self, "target_vectors", None)

        self.extractor = None
        self.target_vectors = {}

        if target_vectors is not None:
            del target_vectors
        if extractor is not None:
            extractor.close()
        else:
            _release_torch_resources()

    @classmethod
    def from_paths(
        cls,
        *,
        model_path: str,
        sae_path: str,
        target_vectors_path: str,
        d_sae: Optional[int] = None,
        device: Optional[str] = None,
        max_length: int = 2048,
        hidden_state_index: int = -2,
        torch_dtype: str = "bfloat16",
        hf_home: Optional[str] = None,
        device_map: Optional[str] = None,
    ) -> "MonaScorer":
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
        target_vectors = cls.load_target_vectors(target_vectors_path)
        return cls(extractor=extractor, target_vectors=target_vectors)

    @classmethod
    def from_eval_datasets(
        cls,
        *,
        model_path: str,
        sae_path: str,
        eval_datasets: Dict[str, Sequence[CanonicalSample]],
        d_sae: Optional[int] = None,
        device: Optional[str] = None,
        max_length: int = 2048,
        hidden_state_index: int = -2,
        torch_dtype: str = "bfloat16",
        hf_home: Optional[str] = None,
        device_map: Optional[str] = None,
        batch_size: int = 8,
        max_eval_samples: Optional[int] = DEFAULT_TARGET_VECTOR_MAX_SAMPLES,
        save_target_vectors: Optional[str] = None,
        devices: Optional[Sequence[str]] = None,
        show_progress: bool = True,
        sample_seed: int = 42,
    ) -> "MonaScorer":
        """Build a MonaScorer by extracting target vectors from eval data.

        This automates the full MONA flow: load model/SAE → extract eval
        features → build target vectors → ready for scoring.

        Parameters
        ----------
        eval_datasets : dict[str, Sequence[CanonicalSample]]
            task_name → eval samples.
        batch_size : int
            Batch size for target vector extraction.
        max_eval_samples : int, optional
            Max samples per eval task.
        save_target_vectors : str, optional
            If provided, save the computed target vectors to this path.
        """
        resolved_devices = [str(item) for item in (devices or []) if str(item)]
        if resolved_devices and len(resolved_devices) > 1 and device_map is None:
            target_vectors = MonaFeatureExtractor.build_target_vectors_multi_device(
                model_path=model_path,
                sae_path=sae_path,
                datasets=eval_datasets,
                devices=resolved_devices,
                d_sae=d_sae,
                max_length=max_length,
                hidden_state_index=hidden_state_index,
                torch_dtype=torch_dtype,
                hf_home=hf_home,
                device_map=None,
                batch_size=batch_size,
                max_samples=max_eval_samples,
                show_progress=show_progress,
                sample_seed=sample_seed,
            )
            extractor = MonaFeatureExtractor.from_paths(
                model_path=model_path,
                sae_path=sae_path,
                d_sae=d_sae,
                device=resolved_devices[0],
                max_length=max_length,
                hidden_state_index=hidden_state_index,
                torch_dtype=torch_dtype,
                hf_home=hf_home,
                device_map=None,
            )
        else:
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
            target_vectors = extractor.build_target_vectors(
                eval_datasets,
                batch_size=batch_size,
                max_samples=max_eval_samples,
                show_progress=show_progress,
                sample_seed=sample_seed,
            )
        if save_target_vectors:
            path = Path(save_target_vectors)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(target_vectors, path)
        return cls(extractor=extractor, target_vectors=target_vectors)

    @staticmethod
    def load_target_vectors(path: str) -> Dict[str, torch.Tensor]:
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError("target_vectors file must contain a dict[str, Tensor]")
        return {
            str(task_name): _coerce_tensor(vector).to(torch.float32)
            for task_name, vector in payload.items()
        }

    def score_sample(self, sample: CanonicalSample) -> MonaScoredRecord:
        feature = self.extractor.extract_sample_feature(sample).detach().to(torch.float32)
        similarities = {
            task_name: generalized_jaccard_similarity(feature, target_vector)
            for task_name, target_vector in self.target_vectors.items()
        }
        return MonaScoredRecord(
            sample_id=sample.sample_id,
            similarities=similarities,
            feature=feature.detach().cpu(),
        )

    def annotate_sample(
        self,
        sample: CanonicalSample,
        *,
        store_feature: bool = False,
        feature_key: str = "feature",
        similarity_key: str = "mona_scores",
    ) -> MonaScoredRecord:
        scored = self.score_sample(sample)
        sample.metadata.extra.setdefault(similarity_key, {}).update(scored.similarities)
        if store_feature:
            sample.metadata.extra[feature_key] = scored.feature.tolist()
        return scored

    def score_dataset(
        self,
        dataset: Sequence[CanonicalSample],
        *,
        annotate_samples: bool = True,
        store_feature: bool = False,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
        progress_interval: int = 100,
    ) -> List[MonaScoredRecord]:
        results = []
        total = len(dataset)
        iterator = _progress_iterable(
            dataset,
            enabled=show_progress,
            total=total,
            desc=progress_desc or f"Scoring on {self.extractor.device}",
            unit="sample",
        )
        for index, sample in enumerate(iterator, start=1):
            if annotate_samples:
                results.append(self.annotate_sample(sample, store_feature=store_feature))
            else:
                results.append(self.score_sample(sample))
            if show_progress and tqdm is None and (index % max(1, progress_interval) == 0 or index == total):
                print(
                    f"[MONA] {progress_desc or self.extractor.device}: {index}/{total} samples processed",
                    flush=True,
                )
        return results

    def score_canonical_jsonl(
        self,
        *,
        input_jsonl: str,
        output_jsonl: str,
        output_pt: Optional[str] = None,
        max_samples: Optional[int] = None,
        store_feature: bool = False,
        show_progress: bool = True,
        progress_desc: Optional[str] = None,
        progress_interval: int = 100,
    ) -> List[MonaScoredRecord]:
        from recipe_sandbox.dataset.loader import CanonicalDatasetLoader
        from recipe_sandbox.schema.io import write_scored_jsonl

        loader = CanonicalDatasetLoader()
        samples = loader.load_all(input_jsonl)
        if max_samples is not None:
            samples = samples[:max_samples]

        if show_progress:
            print(
                f"[MONA] Loaded {len(samples)} samples from {input_jsonl}; starting scoring...",
                flush=True,
            )

        results = self.score_dataset(
            samples,
            annotate_samples=True,
            store_feature=store_feature,
            show_progress=show_progress,
            progress_desc=progress_desc or f"Scoring {Path(input_jsonl).name}",
            progress_interval=progress_interval,
        )
        write_scored_jsonl(output_jsonl, samples)

        if output_pt:
            output_path = Path(output_pt)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "results": [
                        {
                            "id": result.sample_id,
                            "similarities": result.similarities,
                            "feature": result.feature,
                        }
                        for result in results
                    ],
                    "task_names": list(self.target_vectors.keys()),
                    "num_samples": len(results),
                },
                output_path,
            )
        if show_progress:
            print(f"[MONA] Scoring complete → {output_jsonl}", flush=True)
        return results

    def _device(self) -> torch.device:
        return torch.device(self.extractor.device)


def generalized_jaccard_similarity(source: torch.Tensor, target: torch.Tensor) -> float:
    source = source.to(dtype=target.dtype, device=target.device)
    intersection = torch.min(source, target).sum()
    union = torch.max(source, target).sum()
    if union.item() == 0.0:
        return 0.0
    return (intersection / union).item()


def _resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[dtype_name]


def _coerce_tensor(value) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    return torch.tensor(value)


def _infer_runtime_device(model) -> str:
    try:
        parameter = next(model.parameters())
        return str(parameter.device)
    except (AttributeError, StopIteration, TypeError):
        return "cpu"


class _TextListDataset(Dataset):
    """Simple dataset wrapper for batched text feature extraction."""

    def __init__(self, texts: List[str]) -> None:
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


def _partition_named_datasets(
    datasets: Dict[str, Sequence[CanonicalSample]],
    devices: Sequence[str],
) -> List[Tuple[str, List[Tuple[str, Sequence[CanonicalSample]]]]]:
    assignments: List[Tuple[str, List[Tuple[str, Sequence[CanonicalSample]]]]] = [
        (device, []) for device in devices
    ]
    for index, item in enumerate(datasets.items()):
        _, shard = assignments[index % len(assignments)]
        shard.append(item)
    return assignments


def _build_target_vector_worker(payload: dict) -> Dict[str, torch.Tensor]:
    device = payload.get("device", "unknown-device")
    dataset_items = payload.get("dataset_items", [])
    task_names = [task_name for task_name, _ in dataset_items]
    print(
        f"[MONA][{device}] target vector worker started for tasks: {', '.join(task_names)}",
        flush=True,
    )
    extractor = MonaFeatureExtractor.from_paths(
        model_path=payload["model_path"],
        sae_path=payload["sae_path"],
        d_sae=payload.get("d_sae"),
        device=device,
        max_length=payload.get("max_length", 2048),
        hidden_state_index=payload.get("hidden_state_index", -2),
        torch_dtype=payload.get("torch_dtype", "bfloat16"),
        hf_home=payload.get("hf_home"),
        device_map=None,
    )
    result = extractor.build_target_vectors(
        {task_name: samples for task_name, samples in dataset_items},
        batch_size=payload.get("batch_size", 8),
        max_samples=payload.get("max_samples"),
        show_progress=payload.get("show_progress", True),
        sample_seed=payload.get("sample_seed", 42),
    )
    print(
        f"[MONA][{device}] target vector worker finished for {len(result)} task(s)",
        flush=True,
    )
    return result


def _progress_iterable(
    iterable,
    *,
    enabled: bool,
    total: Optional[int],
    desc: str,
    unit: str,
):
    if enabled and tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, unit=unit, leave=False)
    return iterable


def _sample_canonical_samples(
    samples: List[CanonicalSample],
    *,
    max_samples: Optional[int],
    sample_seed: int,
    sample_label: str,
) -> List[CanonicalSample]:
    if max_samples is None or len(samples) <= max_samples:
        return samples

    seed_material = f"{sample_seed}:{sample_label}".encode("utf-8")
    task_seed = int(hashlib.md5(seed_material).hexdigest(), 16) % (2 ** 32)
    rng = random.Random(task_seed)
    selected_indices = sorted(rng.sample(range(len(samples)), max_samples))
    return [samples[index] for index in selected_indices]


def _release_torch_resources(device: Optional[str] = None) -> None:
    """Free GPU/NPU memory caches.

    Args:
        device: If given (e.g. 'cuda:2'), only flush that device's cache.
                If None, flushes all available CUDA + NPU devices.
                In multi-process workers pass the worker's specific device
                to avoid touching unrelated GPUs that may be at OOM.
    """
    gc.collect()

    if torch.cuda.is_available():
        if device is not None and device.startswith("cuda"):
            # Only flush the worker's own device
            try:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
            except Exception:
                pass  # Ignore errors during cleanup
        else:
            # Flush all devices (only for main-process usage)
            for index in range(torch.cuda.device_count()):
                try:
                    with torch.cuda.device(index):
                        torch.cuda.empty_cache()
                        if hasattr(torch.cuda, "ipc_collect"):
                            torch.cuda.ipc_collect()
                except Exception:
                    pass

    npu = getattr(torch, "npu", None)
    if npu and hasattr(npu, "is_available") and npu.is_available():
        if device is not None and device.startswith("npu"):
            try:
                with npu.device(device):
                    if hasattr(npu, "empty_cache"):
                        npu.empty_cache()
            except Exception:
                pass
        else:
            device_count = int(npu.device_count()) if hasattr(npu, "device_count") else 0
            for index in range(device_count):
                try:
                    with npu.device(index):
                        if hasattr(npu, "empty_cache"):
                            npu.empty_cache()
                except Exception:
                    pass
