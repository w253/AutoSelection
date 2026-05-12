"""GPU resource selector for dynamic worker allocation.

Detects available GPUs and their memory usage to select the best devices
for scoring, training, or evaluation workloads. Reuses the MONA-style
worker allocation pattern (one process per device).
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    index: int
    name: str
    total_memory_mb: int
    used_memory_mb: int
    free_memory_mb: int
    utilisation_pct: float

    @property
    def usage_ratio(self) -> float:
        if self.total_memory_mb == 0:
            return 1.0
        return self.used_memory_mb / self.total_memory_mb


def query_gpu_info(timeout: int = 30, retries: int = 2) -> List[GPUInfo]:
    """Query nvidia-smi for GPU memory/utilisation info.

    On some machines the first nvidia-smi call is slow (NVML init can take
    15-30s).  We use a generous *timeout* (default 30s) and retry on
    timeout so that transient driver latency doesn't block the pipeline.
    """
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]

    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode != 0:
                logger.warning("nvidia-smi failed (attempt %d/%d): %s",
                               attempt, retries, result.stderr.strip())
                last_exc = RuntimeError(result.stderr.strip())
                continue

            gpus: List[GPUInfo] = []
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 6:
                    continue
                gpus.append(
                    GPUInfo(
                        index=int(parts[0]),
                        name=parts[1],
                        total_memory_mb=int(parts[2]),
                        used_memory_mb=int(parts[3]),
                        free_memory_mb=int(parts[4]),
                        utilisation_pct=float(parts[5]),
                    )
                )
            return gpus
        except FileNotFoundError:
            logger.warning("nvidia-smi not found. No GPU info available.")
            return []
        except subprocess.TimeoutExpired:
            logger.warning("nvidia-smi timed out after %ds (attempt %d/%d)",
                           timeout, attempt, retries)
            last_exc = subprocess.TimeoutExpired(cmd, timeout)
        except Exception as exc:
            logger.warning("Failed to query GPU info (attempt %d/%d): %s",
                           attempt, retries, exc)
            last_exc = exc

    logger.warning("nvidia-smi failed after %d attempts: %s", retries, last_exc)
    return []


def select_idle_gpus(
    *,
    min_free_mb: int = 8000,
    max_util_pct: float = 30.0,
    max_workers: Optional[int] = None,
) -> List[str]:
    """Select GPUs that are mostly idle (enough free memory + low utilisation).

    Returns a list of CUDA device strings, e.g. ["cuda:0", "cuda:2"].
    """
    gpus = query_gpu_info()
    if not gpus:
        logger.info("No GPUs detected, falling back to CPU.")
        return ["cpu"]

    idle = [
        g
        for g in gpus
        if g.free_memory_mb >= min_free_mb and g.utilisation_pct <= max_util_pct
    ]

    is_fallback = False
    if not idle:
        # Fallback: ignore utilization, sort all GPUs by free memory that satisfy min_free_mb
        target_gpus = [g for g in gpus if g.free_memory_mb >= min_free_mb]
        if not target_gpus:
            # Absolute fallback if even min_free_mb isn't met: pick the one with most free memory
            target_gpus = [max(gpus, key=lambda g: g.free_memory_mb)]
        is_fallback = True
        idle = target_gpus

    # Sort by free memory descending
    idle.sort(key=lambda g: g.free_memory_mb, reverse=True)

    if max_workers is not None:
        idle = idle[:max_workers]

    devices = [f"cuda:{g.index}" for g in idle]
    if is_fallback:
        logger.warning(
            "No strictly idle GPUs found (util <= %.0f%%). Fallback selected %d GPU(s): %s",
            max_util_pct, len(devices),
            ", ".join(f"{d} ({g.free_memory_mb}MB free, {g.utilisation_pct}%% util)" for d, g in zip(devices, idle)),
        )
    else:
        logger.info(
            "Selected %d idle GPU(s): %s",
            len(devices),
            ", ".join(f"{d} ({g.free_memory_mb}MB free)" for d, g in zip(devices, idle)),
        )
    return devices
