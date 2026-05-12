from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import itertools
import math
import multiprocessing as mp
from pathlib import Path
from queue import Empty
from typing import Dict, List, Optional, Type

from recipe_sandbox.pipeline.task_config import TaskConfig
from recipe_sandbox.pipeline.task_manager import TaskManager
from recipe_sandbox.schema.types import CanonicalSample


class ScoringRunner(ABC):
    method_name: str = ""

    def __init__(self, config: TaskConfig, manager: TaskManager) -> None:
        self.config = config
        self.manager = manager

    @abstractmethod
    def run(self) -> None:
        """Execute scoring for the configured method."""


_SCORING_RUNNERS: Dict[str, Type[ScoringRunner]] = {}


def register_scoring_runner(cls: Type[ScoringRunner]) -> Type[ScoringRunner]:
    if not cls.method_name:
        raise ValueError("Scoring runner must define method_name")
    _SCORING_RUNNERS[cls.method_name] = cls
    return cls


def create_scoring_runner(config: TaskConfig, manager: TaskManager) -> ScoringRunner:
    method = config.scoring.method
    runner_cls = _SCORING_RUNNERS.get(method)
    if runner_cls is None:
        supported = ", ".join(sorted(_SCORING_RUNNERS)) or "none"
        raise ValueError(f"Unsupported scoring method: {method}. Supported methods: {supported}")
    return runner_cls(config, manager)


@register_scoring_runner
class MonaScoringRunner(ScoringRunner):
    method_name = "mona"

    def run(self) -> None:
        """Run MONA scoring on all canonical splits.

        If ``target_vectors_path`` is set, loads pre-built target vectors.
        Otherwise, automatically builds target vectors from eval canonical
        data (replicating the ``get_tgt_feature.py`` workflow).
        """
        import torch

        model_cfg = self.config.model
        scoring_cfg = self.config.scoring

        if not model_cfg.sae_path:
            self.manager.log("MONA scoring requires sae_path. Skipping.")
            return

        from recipe_sandbox.scoring.mona import MonaScorer

        scoring_devices = self._resolve_scoring_devices()
        self.manager.log(f"MONA devices: {', '.join(scoring_devices)}")
        self.manager.log(f"MONA workers: {len(scoring_devices)} process(es), one process per device")

        target_vectors_path = model_cfg.target_vectors_path

        if target_vectors_path:
            self.manager.log(f"Loading pre-built target vectors from {target_vectors_path}")
            resolved_target_vectors_path = target_vectors_path
        else:
            eval_files = self.manager.list_canonical("eval")
            if not eval_files:
                self.manager.log(
                    "No target_vectors_path and no eval canonical data. "
                    "Cannot build target vectors. Skipping MONA scoring."
                )
                return

            self.manager.log("Building target vectors from eval canonical data...")
            eval_datasets: Dict[str, List[CanonicalSample]] = {}
            for jsonl_path in eval_files:
                task_name = jsonl_path.stem
                from recipe_sandbox.schema.io import read_jsonl

                samples = list(read_jsonl(str(jsonl_path)))
                eval_datasets[task_name] = samples
                if scoring_cfg.max_eval_samples is not None:
                    self.manager.log(
                        f"  {task_name}: {len(samples)} eval samples, sampling up to {min(len(samples), scoring_cfg.max_eval_samples)} for target vector (seed={scoring_cfg.eval_sample_seed})"
                    )
                else:
                    self.manager.log(f"  {task_name}: {len(samples)} eval samples")

            save_path = str(self.manager.target_vectors_path)
            scorer = None
            try:
                scorer = MonaScorer.from_eval_datasets(
                    model_path=model_cfg.model_path,
                    sae_path=model_cfg.sae_path,
                    eval_datasets=eval_datasets,
                    d_sae=model_cfg.d_sae,
                    device=model_cfg.device,
                    max_length=model_cfg.max_length,
                    hidden_state_index=model_cfg.hidden_state_index,
                    torch_dtype=model_cfg.torch_dtype,
                    hf_home=model_cfg.hf_home,
                    device_map=model_cfg.device_map,
                    batch_size=scoring_cfg.batch_size,
                    max_eval_samples=scoring_cfg.max_eval_samples,
                    save_target_vectors=save_path,
                    devices=scoring_devices,
                    show_progress=scoring_cfg.show_progress,
                    sample_seed=scoring_cfg.eval_sample_seed,
                )
                self.manager.log(
                    f"Target vectors built for {len(scorer.target_vectors)} task(s), "
                    f"saved → {save_path}"
                )
                for task_name, vec in scorer.target_vectors.items():
                    active = torch.count_nonzero(vec).item()
                    total = vec.numel()
                    self.manager.log(
                        f"  {task_name}: active neurons {active}/{total} "
                        f"({active / total * 100:.2f}%)"
                    )
            finally:
                if scorer is not None:
                    scorer.close()
                    self.manager.log("Released MONA eval target-vector builder resources.")

            resolved_target_vectors_path = save_path

        self.manager.log("MonaScorer ready.")

        splits_to_score = ["train"]
        if scoring_cfg.score_eval:
            splits_to_score.append("eval")

        files_to_score = [
            (split, jsonl_path)
            for split in splits_to_score
            for jsonl_path in self.manager.list_canonical(split)
        ]
        if not files_to_score:
            self.manager.log(f"No canonical files found for scoring in splits: {', '.join(splits_to_score)}.")
            return

        scoring_jobs = self._build_scoring_jobs(files_to_score, scoring_devices)

        if len(scoring_devices) > 1 and len(scoring_jobs) > 1 and model_cfg.device_map is None:
            self.manager.log(
                f"Parallel scoring enabled: {len(scoring_jobs)} shard(s) across {len(scoring_devices)} device(s)"
            )
            self._score_parallel_files(
                scoring_jobs=scoring_jobs,
                target_vectors_path=resolved_target_vectors_path,
                devices=scoring_devices,
            )
            return

        if len(scoring_devices) > 1 and model_cfg.device_map is not None:
            self.manager.log("device_map is set; falling back to single-worker scoring.")

        scorer = None
        try:
            scorer = MonaScorer.from_paths(
                model_path=model_cfg.model_path,
                sae_path=model_cfg.sae_path,
                target_vectors_path=resolved_target_vectors_path,
                d_sae=model_cfg.d_sae,
                device=scoring_devices[0],
                max_length=model_cfg.max_length,
                hidden_state_index=model_cfg.hidden_state_index,
                torch_dtype=model_cfg.torch_dtype,
                hf_home=model_cfg.hf_home,
                device_map=model_cfg.device_map,
            )
            self._score_files_with_scorer(files_to_score, scorer)
        finally:
            if scorer is not None:
                scorer.close()
                self.manager.log("Released MONA scorer resources.")

    def _score_parallel_files(
        self,
        *,
        scoring_jobs: List[dict],
        target_vectors_path: str,
        devices: List[str],
    ) -> None:
        model_cfg = self.config.model
        assignments = self._partition_scoring_jobs(scoring_jobs, devices)
        total_shards = len(scoring_jobs)
        completed_shards = 0

        manager = mp.Manager()
        progress_queue = manager.Queue()

        with ProcessPoolExecutor(
            max_workers=len(assignments),
            mp_context=mp.get_context("spawn"),
        ) as executor:
            future_map = {
                executor.submit(
                    _score_mona_job_worker,
                    {
                        "jobs": shard,
                        "device": device,
                        "model_path": model_cfg.model_path,
                        "sae_path": model_cfg.sae_path,
                        "target_vectors_path": target_vectors_path,
                        "d_sae": model_cfg.d_sae,
                        "max_length": model_cfg.max_length,
                        "hidden_state_index": model_cfg.hidden_state_index,
                        "torch_dtype": model_cfg.torch_dtype,
                        "hf_home": model_cfg.hf_home,
                        "store_feature": self.config.scoring.store_feature,
                        "show_progress": self.config.scoring.show_progress,
                        "progress_interval": self.config.scoring.progress_interval,
                        "progress_queue": progress_queue,
                    },
                ): device
                for device, shard in assignments if shard
            }
            worker_results = []
            pending = set(future_map.keys())
            while pending:
                done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                completed_shards = self._drain_scoring_progress_queue(
                    progress_queue,
                    completed_shards=completed_shards,
                    total_shards=total_shards,
                )
                for future in done:
                    worker_results.extend(future.result())

            completed_shards = self._drain_scoring_progress_queue(
                progress_queue,
                completed_shards=completed_shards,
                total_shards=total_shards,
            )

        manager.shutdown()

        self._merge_parallel_scoring_outputs(scoring_jobs, worker_results)

    def _score_files_with_scorer(
        self,
        files_to_score: List[tuple[str, Path]],
        scorer,
        *,
        device: Optional[str] = None,
    ) -> None:
        import torch

        from recipe_sandbox.schema.io import read_jsonl

        scoring_cfg = self.config.scoring
        device_prefix = f"[{device}] " if device else ""

        for split, jsonl_path in files_to_score:
            source_name = jsonl_path.stem
            self.manager.log(f"{device_prefix}Scoring {split}/{source_name}...")
            samples = list(read_jsonl(str(jsonl_path)))

            if scoring_cfg.max_samples is not None:
                samples = samples[:scoring_cfg.max_samples]

            results = scorer.score_dataset(
                samples,
                annotate_samples=True,
                store_feature=scoring_cfg.store_feature,
                show_progress=scoring_cfg.show_progress,
                progress_desc=f"Scoring {split}/{source_name} on {device or scorer.extractor.device}",
                progress_interval=scoring_cfg.progress_interval,
            )

            self.manager.write_scored(split, source_name, samples)

            pt_path = self.manager.scored_path(split, source_name, ext=".pt")
            torch.save(
                {
                    "results": [
                        {
                            "id": r.sample_id,
                            "similarities": r.similarities,
                            "feature": r.feature,
                        }
                        for r in results
                    ],
                    "task_names": list(scorer.target_vectors.keys()),
                    "num_samples": len(results),
                },
                pt_path,
            )
            self.manager.log(f"{device_prefix}Scored {len(results)} samples → {pt_path}")
            if samples:
                self.manager.log(f"{device_prefix}Scored preview:\n{samples[0].pretty}")

    def _build_scoring_jobs(
        self,
        files_to_score: List[tuple[str, Path]],
        devices: List[str],
    ) -> List[dict]:
        jobs: List[dict] = []
        shard_size = self.config.scoring.shard_size
        for split, jsonl_path in files_to_score:
            source_name = jsonl_path.stem
            total_samples = _count_jsonl_records(str(jsonl_path))
            if self.config.scoring.max_samples is not None:
                total_samples = min(total_samples, self.config.scoring.max_samples)
            if total_samples == 0:
                jobs.append(
                    {
                        "split": split,
                        "source_name": source_name,
                        "input_jsonl": str(jsonl_path),
                        "start": 0,
                        "end": 0,
                        "shard_index": 0,
                        "output_jsonl": str(self.manager.scored_shard_path(split, source_name, 0, ext=".jsonl")),
                        "output_pt": str(self.manager.scored_shard_path(split, source_name, 0, ext=".pt")),
                    }
                )
                continue

            resolved_shard_size = shard_size
            if resolved_shard_size is None:
                resolved_shard_size = max(1, math.ceil(total_samples / max(1, len(devices))))

            self.manager.log(
                f"Planning scoring for {split}/{source_name}: {total_samples} samples, shard_size={resolved_shard_size}"
            )

            start = 0
            shard_index = 0
            while start < total_samples:
                end = min(start + resolved_shard_size, total_samples)
                jobs.append(
                    {
                        "split": split,
                        "source_name": source_name,
                        "input_jsonl": str(jsonl_path),
                        "start": start,
                        "end": end,
                        "shard_index": shard_index,
                        "output_jsonl": str(self.manager.scored_shard_path(split, source_name, shard_index, ext=".jsonl")),
                        "output_pt": str(self.manager.scored_shard_path(split, source_name, shard_index, ext=".pt")),
                    }
                )
                start = end
                shard_index += 1
        return jobs

    def _merge_parallel_scoring_outputs(self, scoring_jobs: List[dict], worker_results: List[dict]) -> None:
        import torch

        grouped_jobs: Dict[tuple[str, str], List[dict]] = {}
        for job in scoring_jobs:
            grouped_jobs.setdefault((job["split"], job["source_name"]), []).append(job)

        result_lookup = {
            (item["split"], item["source_name"], item["shard_index"]): item
            for item in worker_results
        }

        for (split, source_name), jobs in grouped_jobs.items():
            ordered_jobs = sorted(jobs, key=lambda item: item["shard_index"])
            final_jsonl = self.manager.scored_path(split, source_name, ext=".jsonl")
            final_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with open(final_jsonl, "w", encoding="utf-8") as out_handle:
                for job in ordered_jobs:
                    shard_path = Path(job["output_jsonl"])
                    if not shard_path.exists():
                        continue
                    with open(shard_path, "r", encoding="utf-8") as in_handle:
                        for line in in_handle:
                            out_handle.write(line)

            combined_results = []
            task_names = None
            num_samples = 0
            for job in ordered_jobs:
                shard_pt = Path(job["output_pt"])
                if not shard_pt.exists():
                    continue
                payload = torch.load(shard_pt, map_location="cpu")
                combined_results.extend(payload.get("results", []))
                if task_names is None:
                    task_names = payload.get("task_names", [])
                num_samples += int(payload.get("num_samples", 0))

            final_pt = self.manager.scored_path(split, source_name, ext=".pt")
            torch.save(
                {
                    "results": combined_results,
                    "task_names": task_names or [],
                    "num_samples": num_samples,
                },
                final_pt,
            )

            preview = None
            for job in ordered_jobs:
                key = (split, source_name, job["shard_index"])
                preview = result_lookup.get(key, {}).get("preview") or preview
            self.manager.log(f"Merged {len(ordered_jobs)} shard(s) → {final_pt}")
            if preview:
                self.manager.log(f"Scored preview:\n{preview}")

    def _drain_scoring_progress_queue(
        self,
        progress_queue,
        *,
        completed_shards: int,
        total_shards: int,
    ) -> int:
        while True:
            try:
                event = progress_queue.get_nowait()
            except Empty:
                break
            completed_shards += 1
            self.manager.log(
                "Shard progress: "
                f"{completed_shards}/{total_shards} completed "
                f"({event['device']} {event['split']}/{event['source_name']} "
                f"[{event['start']}:{event['end']}])"
            )
        return completed_shards

    def _resolve_scoring_devices(self) -> List[str]:
        import torch

        model_cfg = self.config.model
        scoring_cfg = self.config.scoring

        if scoring_cfg.devices:
            return [str(device) for device in scoring_cfg.devices]

        if model_cfg.device and model_cfg.device not in {"cuda", "npu"}:
            return [model_cfg.device]

        if model_cfg.device in {None, "npu"} and _npu_is_available(torch):
            npu_count = _npu_device_count(torch)
            if npu_count > 1:
                workers = scoring_cfg.parallel_workers or npu_count
                workers = max(1, min(workers, npu_count))
                return [f"npu:{index}" for index in range(workers)]
            return ["npu:0"]

        if model_cfg.device in {None, "cuda"} and torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                workers = scoring_cfg.parallel_workers or gpu_count
                workers = max(1, min(workers, gpu_count))
                return [f"cuda:{index}" for index in range(workers)]
            return ["cuda:0"]

        return [model_cfg.device or "cpu"]

    def _partition_scoring_jobs(
        self,
        scoring_jobs: List[dict],
        devices: List[str],
    ) -> List[tuple[str, List[dict]]]:
        assignments: List[tuple[str, List[dict]]] = [
            (device, []) for device in devices
        ]
        for index, item in enumerate(scoring_jobs):
            _, shard = assignments[index % len(assignments)]
            shard.append(item)
        return assignments


def _score_mona_job_worker(payload: dict) -> List[dict]:
    import torch

    from recipe_sandbox.schema.io import write_scored_jsonl
    from recipe_sandbox.scoring.mona import MonaScorer

    scorer = MonaScorer.from_paths(
        model_path=payload["model_path"],
        sae_path=payload["sae_path"],
        target_vectors_path=payload["target_vectors_path"],
        d_sae=payload.get("d_sae"),
        device=payload.get("device"),
        max_length=payload.get("max_length", 2048),
        hidden_state_index=payload.get("hidden_state_index", -2),
        torch_dtype=payload.get("torch_dtype", "bfloat16"),
        hf_home=payload.get("hf_home"),
        device_map=None,
    )

    worker_results: List[dict] = []
    progress_queue = payload.get("progress_queue")
    for job in payload["jobs"]:
        device = payload.get("device", "unknown-device")
        print(
            f"[MONA][{device}] shard {job['split']}/{job['source_name']} "
            f"[{job['start']}:{job['end']}] started",
            flush=True,
        )
        samples = _read_jsonl_slice(job["input_jsonl"], job["start"], job["end"])
        results = scorer.score_dataset(
            samples,
            annotate_samples=True,
            store_feature=payload.get("store_feature", False),
            show_progress=payload.get("show_progress", True),
            progress_desc=(
                f"{device} {job['split']}/{job['source_name']} "
                f"[{job['start']}:{job['end']}]"
            ),
            progress_interval=payload.get("progress_interval", 100),
        )
        write_scored_jsonl(job["output_jsonl"], samples)
        Path(job["output_pt"]).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "results": [
                    {
                        "id": r.sample_id,
                        "similarities": r.similarities,
                        "feature": r.feature,
                    }
                    for r in results
                ],
                "task_names": list(scorer.target_vectors.keys()),
                "num_samples": len(results),
            },
            job["output_pt"],
        )
        worker_results.append(
            {
                "split": job["split"],
                "source_name": job["source_name"],
                "shard_index": job["shard_index"],
                "count": len(results),
                "device": payload.get("device"),
                "preview": samples[0].pretty if samples else None,
            }
        )
        if progress_queue is not None:
            progress_queue.put(
                {
                    "device": device,
                    "split": job["split"],
                    "source_name": job["source_name"],
                    "start": job["start"],
                    "end": job["end"],
                }
            )
        print(
            f"[MONA][{device}] shard {job['split']}/{job['source_name']} "
            f"[{job['start']}:{job['end']}] finished ({len(results)} samples)",
            flush=True,
        )
    return worker_results


def _read_jsonl_slice(path: str, start: int, end: int) -> List[CanonicalSample]:
    from recipe_sandbox.schema.io import read_jsonl

    if end <= start:
        return []
    return list(itertools.islice(read_jsonl(path), start, end))


def _count_jsonl_records(path: str) -> int:
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


@register_scoring_runner
class LessScoringRunner(ScoringRunner):
    """LESS influence scoring runner.

    Reads canonical samples, extracts representations or gradients via
    a ``LessRepresentationExtractor`` or ``LessGradientExtractor``,
    computes influence scores against validation samples using
    ``LessScorer``, and writes scored output.
    """

    method_name = "less"

    def run(self) -> None:
        import torch

        from recipe_sandbox.schema.io import read_jsonl, write_scored_jsonl
        from recipe_sandbox.scoring.less import LessRepresentationExtractor, LessScorer

        model_cfg = self.config.model
        scoring_cfg = self.config.scoring

        # Load model and tokenizer
        self.manager.log("Loading model for LESS scoring...")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        torch_dtype = dtype_map.get(model_cfg.torch_dtype, torch.bfloat16)
        device = model_cfg.device or "cpu"

        tokenizer = AutoTokenizer.from_pretrained(
            model_cfg.model_path,
            cache_dir=model_cfg.hf_home,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg.model_path,
            torch_dtype=torch_dtype,
            cache_dir=model_cfg.hf_home,
            device_map=model_cfg.device_map,
            trust_remote_code=True,
        )
        if model_cfg.device_map is None:
            model = model.to(device)

        extractor = LessRepresentationExtractor(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=model_cfg.max_length,
        )

        # Load validation samples
        eval_files = self.manager.list_canonical("eval")
        if not eval_files:
            self.manager.log("No eval canonical data for LESS validation. Skipping.")
            return

        validation_samples: list[CanonicalSample] = []
        for jsonl_path in eval_files:
            samples = list(read_jsonl(str(jsonl_path)))
            if scoring_cfg.max_eval_samples is not None:
                samples = samples[: scoring_cfg.max_eval_samples]
            validation_samples.extend(samples)
            self.manager.log(f"  eval: {jsonl_path.stem}: {len(samples)} samples")

        self.manager.log(f"LESS validation samples: {len(validation_samples)}")
        scorer = LessScorer()

        # Score train (and optionally eval) splits
        splits_to_score = ["train"]
        if scoring_cfg.score_eval:
            splits_to_score.append("eval")

        for split in splits_to_score:
            for jsonl_path in self.manager.list_canonical(split):
                source_name = jsonl_path.stem
                self.manager.log(f"LESS scoring {split}/{source_name}...")
                train_samples = list(read_jsonl(str(jsonl_path)))
                if scoring_cfg.max_samples is not None:
                    train_samples = train_samples[: scoring_cfg.max_samples]

                result = scorer.score_datasets(
                    train_dataset=train_samples,
                    validation_dataset=validation_samples,
                    info_extractor=extractor,
                    sample_ids=[s.sample_id for s in train_samples],
                )

                score_map = dict(zip(result.sample_ids, result.values.tolist()))
                for sample in train_samples:
                    score = score_map.get(sample.sample_id)
                    if score is not None:
                        sample.metadata.extra.setdefault("less", {})["influence_score"] = float(score)

                self.manager.write_scored(split, source_name, train_samples)
                self.manager.log(f"LESS scored {len(train_samples)} samples -> {split}/{source_name}")


def _npu_is_available(torch_module) -> bool:
    npu = getattr(torch_module, "npu", None)
    return bool(npu) and hasattr(npu, "is_available") and bool(npu.is_available())


def _npu_device_count(torch_module) -> int:
    npu = getattr(torch_module, "npu", None)
    if not npu or not hasattr(npu, "device_count"):
        return 0
    return int(npu.device_count())
