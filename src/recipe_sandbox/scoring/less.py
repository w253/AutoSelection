from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import torch
import torch.nn.functional as F

from recipe_sandbox.operators.helpers import sample_to_text
from recipe_sandbox.scoring.base import ScoringBatchResult, normalize_weights
from recipe_sandbox.schema.types import CanonicalSample


@dataclass
class LessInfoLibrary:
    sample_ids: list[str]
    infos: torch.Tensor
    output_dir: Optional[str] = None
    prefix: Optional[str] = None


class RandomProjector:
    def __init__(
        self,
        grad_dim: int,
        proj_dim: int,
        *,
        seed: int = 0,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> None:
        self.grad_dim = grad_dim
        self.proj_dim = proj_dim
        self.seed = seed
        self.dtype = dtype
        self.device = torch.device(device)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        self.matrix = torch.randint(
            low=0,
            high=2,
            size=(grad_dim, proj_dim),
            generator=generator,
            dtype=torch.int8,
        ).to(torch.float32)
        self.matrix = self.matrix.mul_(2.0).sub_(1.0).to(dtype=self.dtype, device=self.device)

    def project(self, batch: torch.Tensor, model_id: int = 0) -> torch.Tensor:
        del model_id
        inputs = batch.to(device=self.device, dtype=self.dtype)
        return torch.matmul(inputs, self.matrix)


class LessInfoLibraryBuilder:
    def __init__(
        self,
        info_extractor,
        output_dir: str,
        *,
        prefix: str = "reps",
        save_interval: int = 160,
        normalize_merged: bool = True,
        projected_dims: Optional[Sequence[int]] = None,
        projector_factory: Optional[Callable[..., Any]] = None,
        projection_seed: int = 0,
        projection_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.info_extractor = info_extractor
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.save_interval = save_interval
        self.normalize_merged = normalize_merged
        self.projected_dims = list(projected_dims or [])
        self.projector_factory = projector_factory
        self.projection_seed = projection_seed
        self.projection_dtype = projection_dtype
        self._projectors: Dict[int, Any] = {}

    def build(
        self,
        dataset: Sequence[CanonicalSample],
        *,
        sample_ids: Optional[Sequence[str]] = None,
        max_samples: Optional[int] = None,
    ) -> LessInfoLibrary:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        ids = list(sample_ids) if sample_ids is not None else [sample.sample_id for sample in dataset]
        collected_infos = []
        collected_ids = []
        buffer_infos = []
        buffer_ids = []

        total = len(dataset) if max_samples is None else min(len(dataset), max_samples)
        for index, sample in enumerate(dataset[:total], start=1):
            info = self._extract_sample_info(sample).detach().cpu().to(torch.float32)
            buffer_infos.append(info)
            buffer_ids.append(ids[index - 1])
            collected_infos.append(info)
            collected_ids.append(ids[index - 1])

            if index % self.save_interval == 0:
                self._save_chunk(buffer_infos, buffer_ids, index)
                buffer_infos = []
                buffer_ids = []

        if buffer_infos:
            self._save_chunk(buffer_infos, buffer_ids, total)

        merged = self.merge_chunks(normalized=self.normalize_merged)
        self.merge_chunks(normalized=False)
        for projected_dim in self.projected_dims:
            self.merge_projected_chunks(projected_dim, normalized=self.normalize_merged)
            self.merge_projected_chunks(projected_dim, normalized=False)
        self._save_sample_ids(collected_ids)
        self._save_manifest(total_samples=len(collected_ids))

        return LessInfoLibrary(
            sample_ids=collected_ids,
            infos=merged,
            output_dir=str(self.output_dir),
            prefix=self.prefix,
        )

    def merge_chunks(self, *, normalized: bool) -> torch.Tensor:
        chunks = self._load_chunk_files()
        merged = []
        for chunk_path in chunks:
            data = torch.load(chunk_path, map_location="cpu")
            if normalized:
                data = F.normalize(data.to(torch.float32), dim=1)
            else:
                data = data.to(torch.float32)
            merged.append(data)

        if merged:
            output = torch.cat(merged, dim=0)
        else:
            output = torch.empty((0, 0), dtype=torch.float32)

        output_name = "all_orig.pt" if normalized else "all_unormalized.pt"
        torch.save(output, self.output_dir / output_name)
        return output

    def merge_projected_chunks(self, projected_dim: int, *, normalized: bool) -> torch.Tensor:
        projected_dir = self._projected_output_dir(projected_dim)
        chunks = self._load_chunk_files(base_dir=projected_dir)
        merged = []
        for chunk_path in chunks:
            data = torch.load(chunk_path, map_location="cpu")
            if normalized:
                data = F.normalize(data.to(torch.float32), dim=1)
            else:
                data = data.to(torch.float32)
            merged.append(data)

        if merged:
            output = torch.cat(merged, dim=0)
        else:
            output = torch.empty((0, 0), dtype=torch.float32)

        output_name = "all_orig.pt" if normalized else "all_unormalized.pt"
        torch.save(output, projected_dir / output_name)
        return output

    def load_library(self, *, normalized: bool = True) -> LessInfoLibrary:
        infos = self.load_info_tensor(self.output_dir, normalized=normalized)
        sample_ids = self.load_sample_ids(self.output_dir)
        return LessInfoLibrary(
            sample_ids=sample_ids,
            infos=infos,
            output_dir=str(self.output_dir),
            prefix=self.prefix,
        )

    def load_projected_library(self, projected_dim: int, *, normalized: bool = True) -> LessInfoLibrary:
        projected_dir = self._projected_output_dir(projected_dim)
        infos = self.load_info_tensor(projected_dir, normalized=normalized)
        sample_ids = self.load_sample_ids(projected_dir)
        return LessInfoLibrary(
            sample_ids=sample_ids,
            infos=infos,
            output_dir=str(projected_dir),
            prefix=self.prefix,
        )

    @staticmethod
    def load_info_tensor(output_dir: str | Path, *, normalized: bool = True) -> torch.Tensor:
        base = Path(output_dir)
        filename = "all_orig.pt" if normalized else "all_unormalized.pt"
        return torch.load(base / filename, map_location="cpu")

    @staticmethod
    def load_sample_ids(output_dir: str | Path) -> list[str]:
        path = Path(output_dir) / "sample_ids.json"
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def _extract_sample_info(self, sample: CanonicalSample) -> torch.Tensor:
        if hasattr(self.info_extractor, "extract_sample_info"):
            return self.info_extractor.extract_sample_info(sample)
        if hasattr(self.info_extractor, "extract_sample_representation"):
            return self.info_extractor.extract_sample_representation(sample)
        raise TypeError("info_extractor must expose extract_sample_info or extract_sample_representation")

    def _save_chunk(self, infos: Sequence[torch.Tensor], sample_ids: Sequence[str], count: int) -> None:
        chunk = torch.stack(list(infos), dim=0)
        torch.save(chunk, self.output_dir / f"{self.prefix}-{count}.pt")
        metadata_path = self.output_dir / f"{self.prefix}-{count}.json"
        metadata_path.write_text(json.dumps(list(sample_ids), ensure_ascii=False), encoding="utf-8")
        self._save_projected_chunks(chunk, sample_ids, count)

    def _load_chunk_files(self, base_dir: Optional[Path] = None) -> list[Path]:
        current_dir = base_dir or self.output_dir
        files = sorted(
            current_dir.glob(f"{self.prefix}-*.pt"),
            key=lambda path: int(path.stem.split("-")[-1]),
        )
        return files

    def _save_sample_ids(self, sample_ids: Sequence[str]) -> None:
        path = self.output_dir / "sample_ids.json"
        path.write_text(json.dumps(list(sample_ids), ensure_ascii=False), encoding="utf-8")
        for projected_dim in self.projected_dims:
            projected_dir = self._projected_output_dir(projected_dim)
            projected_dir.mkdir(parents=True, exist_ok=True)
            (projected_dir / "sample_ids.json").write_text(
                json.dumps(list(sample_ids), ensure_ascii=False),
                encoding="utf-8",
            )

    def _save_manifest(self, *, total_samples: int) -> None:
        manifest = {
            "prefix": self.prefix,
            "save_interval": self.save_interval,
            "normalize_merged": self.normalize_merged,
            "total_samples": total_samples,
            "projected_dims": self.projected_dims,
        }
        (self.output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        for projected_dim in self.projected_dims:
            projected_dir = self._projected_output_dir(projected_dim)
            projected_dir.mkdir(parents=True, exist_ok=True)
            (projected_dir / "manifest.json").write_text(
                json.dumps({**manifest, "projected_dim": projected_dim}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _save_projected_chunks(self, chunk: torch.Tensor, sample_ids: Sequence[str], count: int) -> None:
        if not self.projected_dims:
            return
        self._ensure_projectors(chunk.shape[1])
        batch = chunk.to(dtype=self.projection_dtype)
        for projected_dim, projector in self._projectors.items():
            projected_dir = self._projected_output_dir(projected_dim)
            projected_dir.mkdir(parents=True, exist_ok=True)
            projected_chunk = projector.project(batch, model_id=0).detach().cpu().to(torch.float32)
            torch.save(projected_chunk, projected_dir / f"{self.prefix}-{count}.pt")
            (projected_dir / f"{self.prefix}-{count}.json").write_text(
                json.dumps(list(sample_ids), ensure_ascii=False),
                encoding="utf-8",
            )

    def _ensure_projectors(self, grad_dim: int) -> None:
        if self._projectors:
            return
        for offset, projected_dim in enumerate(self.projected_dims):
            if self.projector_factory is not None:
                projector = self.projector_factory(
                    grad_dim=grad_dim,
                    proj_dim=projected_dim,
                    seed=self.projection_seed + offset,
                    dtype=self.projection_dtype,
                )
            else:
                projector = RandomProjector(
                    grad_dim=grad_dim,
                    proj_dim=projected_dim,
                    seed=self.projection_seed + offset,
                    dtype=self.projection_dtype,
                )
            self._projectors[projected_dim] = projector

    def _projected_output_dir(self, projected_dim: int) -> Path:
        return self.output_dir / f"dim{projected_dim}"


class LessRepresentationExtractor:
    def __init__(
        self,
        model,
        tokenizer,
        *,
        device: Optional[str] = None,
        max_length: int = 2048,
        normalize: bool = True,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or self._infer_device(model)
        self.max_length = max_length
        self.normalize = normalize

    def extract_sample_info(self, sample: CanonicalSample) -> torch.Tensor:
        return self.extract_sample_representation(sample)

    def extract_dataset_infos(self, dataset: Sequence[CanonicalSample]) -> torch.Tensor:
        return self.extract_dataset_representations(dataset)

    def extract_sample_representation(self, sample: CanonicalSample) -> torch.Tensor:
        return self.extract_text_representation(sample_to_text(sample))

    def extract_dataset_representations(self, dataset: Sequence[CanonicalSample]) -> torch.Tensor:
        representations = [self.extract_sample_representation(sample) for sample in dataset]
        if not representations:
            return torch.empty((0, 0), dtype=torch.float32)
        return torch.stack(representations, dim=0)

    def extract_text_representation(self, text: str) -> torch.Tensor:
        batch = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        batch = self._move_to_device(batch, self.device)
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                representation = outputs.pooler_output.squeeze(0)
            else:
                hidden_states = outputs.hidden_states[-1]
                position = attention_mask.sum(dim=1) - 1
                batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
                representation = hidden_states[batch_indices, position].squeeze(0)

        representation = representation.to(torch.float32)
        if self.normalize:
            representation = F.normalize(representation, dim=0)
        return representation

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


class LessGradientExtractor:
    def __init__(
        self,
        model,
        tokenizer,
        *,
        device: Optional[str] = None,
        max_length: int = 2048,
        gradient_type: str = "sgd",
        adam_optimizer_state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        normalize: bool = True,
        projection_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        parameter_filter: Optional[Callable[[str, torch.nn.Parameter], bool]] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or self._infer_device(model)
        self.max_length = max_length
        self.gradient_type = gradient_type
        self.adam_optimizer_state = adam_optimizer_state
        self.normalize = normalize
        self.projection_fn = projection_fn
        self.parameter_filter = parameter_filter

    def extract_sample_info(self, sample: CanonicalSample) -> torch.Tensor:
        return self.extract_sample_gradient(sample)

    def extract_dataset_infos(self, dataset: Sequence[CanonicalSample]) -> torch.Tensor:
        gradients = [self.extract_sample_gradient(sample) for sample in dataset]
        if not gradients:
            return torch.empty((0, 0), dtype=torch.float32)
        return torch.stack(gradients, dim=0)

    def extract_sample_gradient(self, sample: CanonicalSample) -> torch.Tensor:
        return self.extract_text_gradient(sample_to_text(sample))

    def extract_text_gradient(self, text: str) -> torch.Tensor:
        batch = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        batch = self._move_to_device(batch, self.device)
        batch = self._ensure_labels(batch)

        self.model.zero_grad()
        outputs = self.model(**batch)
        loss = outputs.loss
        loss.backward()

        vector = self._vectorize_gradients()
        self.model.zero_grad()

        if self.projection_fn is not None:
            vector = self.projection_fn(vector)

        vector = vector.to(torch.float32)
        if self.normalize and vector.numel() > 0:
            vector = F.normalize(vector, dim=0)
        return vector.detach().cpu()

    def _vectorize_gradients(self) -> torch.Tensor:
        named_params = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.grad is not None and self._include_parameter(name, parameter)
        ]
        if not named_params:
            return torch.empty((0,), dtype=torch.float32, device=self._torch_device())

        if self.gradient_type == "adam":
            if self.adam_optimizer_state is None:
                raise ValueError("adam_optimizer_state is required when gradient_type='adam'")
            return self._vectorize_adam_gradients(named_params)

        vectors = []
        for _, parameter in named_params:
            grad = parameter.grad.detach()
            if self.gradient_type == "sign":
                grad = torch.sign(grad)
            vectors.append(grad.reshape(-1))
        return torch.cat(vectors, dim=0)

    def _vectorize_adam_gradients(self, named_params) -> torch.Tensor:
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-08
        vectors = []

        for name, parameter in named_params:
            state = self.adam_optimizer_state.get(name)
            if state is None:
                raise KeyError(f"Missing Adam optimizer state for parameter '{name}'")
            avg = state["exp_avg"].to(parameter.grad.device)
            avg_sq = state["exp_avg_sq"].to(parameter.grad.device)
            grad = parameter.grad.detach()
            updated_avg = beta1 * avg + (1 - beta1) * grad
            updated_avg_sq = beta2 * avg_sq + (1 - beta2) * grad.pow(2)
            vectors.append((updated_avg / torch.sqrt(updated_avg_sq + eps)).reshape(-1))

        return torch.cat(vectors, dim=0)

    def _include_parameter(self, name: str, parameter: torch.nn.Parameter) -> bool:
        if self.parameter_filter is not None:
            return bool(self.parameter_filter(name, parameter))
        return parameter.requires_grad

    def _ensure_labels(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if "labels" in batch:
            return batch
        output = dict(batch)
        output["labels"] = batch["input_ids"]
        return output

    def _infer_device(self, model) -> str:
        try:
            parameter = next(model.parameters())
            return str(parameter.device)
        except (AttributeError, StopIteration, TypeError):
            return "cpu"

    def _torch_device(self) -> torch.device:
        return torch.device(self.device)

    def _move_to_device(self, batch, device: str):
        if hasattr(batch, "to"):
            return batch.to(device)
        if isinstance(batch, dict):
            return {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }
        return batch


class LessScorer:
    def __init__(
        self,
        *,
        checkpoint_weights: Optional[Sequence[float]] = None,
        num_subtasks: Optional[int] = None,
        subtask_reduce: str = "mean",
        task_reduce: str = "max",
    ) -> None:
        self.checkpoint_weights = checkpoint_weights
        self.num_subtasks = num_subtasks
        self.subtask_reduce = subtask_reduce
        self.task_reduce = task_reduce

    def calculate_influence_matrix(
        self,
        training_info: torch.Tensor,
        validation_info: torch.Tensor,
    ) -> torch.Tensor:
        return torch.matmul(training_info.to(torch.float32), validation_info.to(torch.float32).transpose(0, 1))

    def score(
        self,
        training_info: torch.Tensor,
        validation_info: torch.Tensor,
        *,
        num_subtasks: Optional[int] = None,
        subtask_reduce: Optional[str] = None,
        task_reduce: Optional[str] = None,
    ) -> torch.Tensor:
        influence = self.calculate_influence_matrix(training_info, validation_info)
        return self.aggregate_validation_scores(
            influence,
            num_subtasks=num_subtasks,
            subtask_reduce=subtask_reduce,
            task_reduce=task_reduce,
        )

    def score_checkpoints(
        self,
        training_infos: Sequence[torch.Tensor],
        validation_infos: Sequence[torch.Tensor],
        *,
        checkpoint_weights: Optional[Sequence[float]] = None,
        num_subtasks: Optional[int] = None,
        subtask_reduce: Optional[str] = None,
        task_reduce: Optional[str] = None,
    ) -> torch.Tensor:
        if len(training_infos) != len(validation_infos):
            raise ValueError("training_infos and validation_infos must have the same number of checkpoints")
        if not training_infos:
            return torch.empty((0,), dtype=torch.float32)

        weights = normalize_weights(checkpoint_weights or self.checkpoint_weights or [1.0] * len(training_infos))
        influence = None
        for weight, train_info, valid_info in zip(weights, training_infos, validation_infos):
            current = weight * self.calculate_influence_matrix(train_info, valid_info)
            influence = current if influence is None else influence + current
        return self.aggregate_validation_scores(
            influence,
            num_subtasks=num_subtasks,
            subtask_reduce=subtask_reduce,
            task_reduce=task_reduce,
        )

    def score_from_saved_tensors(
        self,
        training_info_path: str,
        validation_info_path: str,
        *,
        num_subtasks: Optional[int] = None,
        subtask_reduce: Optional[str] = None,
        task_reduce: Optional[str] = None,
    ) -> torch.Tensor:
        training_info = torch.load(training_info_path, map_location="cpu")
        validation_info = torch.load(validation_info_path, map_location="cpu")
        return self.score(
            training_info=training_info,
            validation_info=validation_info,
            num_subtasks=num_subtasks,
            subtask_reduce=subtask_reduce,
            task_reduce=task_reduce,
        )

    def score_datasets(
        self,
        train_dataset: Sequence[CanonicalSample],
        validation_dataset: Sequence[CanonicalSample],
        info_extractor,
        *,
        sample_ids: Optional[Sequence[str]] = None,
        num_subtasks: Optional[int] = None,
    ) -> ScoringBatchResult:
        train_representations = self._extract_dataset_infos(info_extractor, train_dataset)
        validation_representations = self._extract_dataset_infos(info_extractor, validation_dataset)
        scores = self.score(
            train_representations,
            validation_representations,
            num_subtasks=num_subtasks,
        )
        ids = list(sample_ids) if sample_ids is not None else [sample.sample_id for sample in train_dataset]
        return ScoringBatchResult(sample_ids=ids, values=scores)

    def aggregate_validation_scores(
        self,
        influence_scores: torch.Tensor,
        *,
        num_subtasks: Optional[int] = None,
        subtask_reduce: Optional[str] = None,
        task_reduce: Optional[str] = None,
    ) -> torch.Tensor:
        if influence_scores.ndim != 2:
            raise ValueError("influence_scores must have shape [n_train, n_validation]")

        current_num_subtasks = num_subtasks or self.num_subtasks
        if current_num_subtasks is None or current_num_subtasks <= 1:
            return self._reduce(influence_scores, dim=1, mode=task_reduce or self.task_reduce)

        if influence_scores.shape[1] % current_num_subtasks != 0:
            raise ValueError("validation count must be divisible by num_subtasks")

        examples_per_subtask = influence_scores.shape[1] // current_num_subtasks
        reshaped = influence_scores.reshape(influence_scores.shape[0], current_num_subtasks, examples_per_subtask)
        reduced_within_subtask = self._reduce(
            reshaped,
            dim=2,
            mode=subtask_reduce or self.subtask_reduce,
        )
        return self._reduce(
            reduced_within_subtask,
            dim=1,
            mode=task_reduce or self.task_reduce,
        )

    def _reduce(self, tensor: torch.Tensor, *, dim: int, mode: str) -> torch.Tensor:
        if mode == "sum":
            return tensor.sum(dim=dim)
        if mode == "max":
            return tensor.max(dim=dim).values
        return tensor.mean(dim=dim)

    def _extract_dataset_infos(self, info_extractor, dataset: Sequence[CanonicalSample]) -> torch.Tensor:
        if hasattr(info_extractor, "extract_dataset_infos"):
            return info_extractor.extract_dataset_infos(dataset)
        if hasattr(info_extractor, "extract_dataset_representations"):
            return info_extractor.extract_dataset_representations(dataset)
        raise TypeError("info_extractor must expose extract_dataset_infos or extract_dataset_representations")