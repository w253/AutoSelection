"""Device resource cleanup utilities (CUDA + NPU).

Call ``release_all_device_memory()`` before any device-intensive stage to
ensure stale contexts, cached tensors, and zombie processes don't block
device memory.  This is especially important in long-running Agent loops
where SAE ingest, LoRA training, and vLLM inference alternate on shared
accelerators.
"""

from __future__ import annotations

import gc
import logging
import os
import re
import signal
import subprocess
import time
from typing import List, Optional, Set

logger = logging.getLogger(__name__)


def _npu_cleanup_enabled() -> bool:
    """Whether explicit NPU cache cleanup is enabled.

    Some Ascend environments fail inside ``torch.npu.empty_cache()`` because
    calling into ACL/TBE re-initializes compiler components that are not fully
    available in the current runtime. In those environments, attempting cleanup
    is worse than leaving the live process untouched.
    """
    value = os.getenv("RECIPE_SANDBOX_ENABLE_NPU_CLEANUP", "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


# ---------------------------------------------------------------------------
#  Detect backend
# ---------------------------------------------------------------------------

def _detect_backend() -> str:
    """Return 'npu', 'cuda', or 'none'."""
    try:
        import torch
    except ImportError:
        return "none"
    npu = getattr(torch, "npu", None)
    if npu and hasattr(npu, "is_available") and npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "none"


# ---------------------------------------------------------------------------
#  Current-process memory release
# ---------------------------------------------------------------------------

def release_all_gpu_memory(
    *,
    wait_seconds: float = 2.0,
    kill_zombies: bool = False,
) -> None:
    """Release device memory held by the **current** process.

    Handles both CUDA and NPU backends transparently.

    ``kill_zombies`` is intentionally ignored: sweeping all child processes of
    the current Python process is unsafe on Ascend because TBE/torch_npu may
    spawn manager/helper processes that are still required by the live parent.
    """
    backend = _detect_backend()
    if backend == "none":
        logger.debug("No accelerator available, skipping cleanup.")
        return

    import torch

    gc.collect()

    if backend == "npu":
        npu = torch.npu
        device_count = npu.device_count()
        if not _npu_cleanup_enabled():
            logger.info(
                "NPU cleanup: skipping torch.npu.empty_cache() across %d device(s) "
                "(set RECIPE_SANDBOX_ENABLE_NPU_CLEANUP=1 to enable).",
                device_count,
            )
        else:
            logger.info("NPU cleanup: releasing memory across %d device(s) ...", device_count)
            for i in range(device_count):
                try:
                    with npu.device(i):
                        npu.empty_cache()
                except Exception as exc:
                    logger.warning("Failed to clean npu:%d: %s", i, exc)
    else:
        device_count = torch.cuda.device_count()
        logger.info("CUDA cleanup: releasing memory across %d device(s) ...", device_count)
        for i in range(device_count):
            try:
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.reset_accumulated_memory_stats()
            except Exception as exc:
                logger.warning("Failed to clean cuda:%d: %s", i, exc)

    gc.collect()

    if kill_zombies:
        logger.warning(
            "kill_zombies=True requested, but broad child-process cleanup is "
            "disabled because it can terminate live Ascend/TBE helpers."
        )

    if wait_seconds > 0:
        time.sleep(wait_seconds)

    logger.info("Device cleanup done (%s).", backend)


def release_model_from_memory(model_obj: object, label: str = "model") -> None:
    """Delete a model object and aggressively free its device memory."""
    try:
        import torch
    except ImportError:
        return

    if model_obj is None:
        return

    logger.info("Releasing %s from device memory ...", label)

    try:
        if hasattr(model_obj, "cpu"):
            model_obj.cpu()
    except Exception:
        pass

    del model_obj
    gc.collect()

    backend = _detect_backend()
    if backend == "npu":
        if _npu_cleanup_enabled():
            try:
                torch.npu.empty_cache()
            except Exception as exc:
                logger.warning("Failed to clean NPU cache while releasing %s: %s", label, exc)
    elif backend == "cuda":
        torch.cuda.empty_cache()

    logger.info("  %s released.", label)

# ---------------------------------------------------------------------------
#  Kill stale processes on specific NPU devices (via npu-smi)
# ---------------------------------------------------------------------------

_PROC_MEM_RE = re.compile(r"Process id\s*:\s*(\d+)")


def _query_npu_device_pids(device_id: int) -> Set[int]:
    """Return PIDs holding HBM on a specific NPU device via npu-smi."""
    try:
        result = subprocess.run(
            ["npu-smi", "info", "-t", "proc-mem", "-i", str(device_id)],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return set()
        return {int(m.group(1)) for m in _PROC_MEM_RE.finditer(result.stdout)}
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("npu-smi proc-mem query for device %d failed: %s", device_id, exc)
        return set()


def kill_stale_npu_processes(device_ids: List[int]) -> None:
    """Find and kill ALL processes holding HBM on the given NPU devices.

    Skips the current process. Used before launching training/eval
    subprocesses to ensure target NPU devices are free.
    """
    current_pid = os.getpid()
    pids_to_kill: Set[int] = set()

    for dev_id in device_ids:
        pids = _query_npu_device_pids(dev_id)
        pids_to_kill.update(pids)

    pids_to_kill.discard(current_pid)

    if not pids_to_kill:
        return

    logger.warning(
        "Killing %d stale process(es) on NPU devices %s: %s",
        len(pids_to_kill), device_ids, sorted(pids_to_kill),
    )

    for pid in pids_to_kill:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    time.sleep(2)

    for pid in pids_to_kill:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    time.sleep(1)

    # Verify cleanup
    remaining: Set[int] = set()
    for dev_id in device_ids:
        pids = _query_npu_device_pids(dev_id)
        pids.discard(current_pid)
        remaining.update(pids)

    if remaining:
        logger.error(
            "Failed to kill %d process(es) on NPU devices %s: %s",
            len(remaining), device_ids, sorted(remaining),
        )
    else:
        logger.info("NPU devices %s are now free.", device_ids)


def ensure_devices_free(device_ids: List[int], device_type: str) -> None:
    """Ensure target devices have no stale processes before launching workloads.

    Parameters
    ----------
    device_ids:
        List of device indices to clean (e.g. [0, 1, 2, 3]).
    device_type:
        'npu' or 'gpu'.
    """
    if device_type == "npu":
        kill_stale_npu_processes(device_ids)
    # CUDA: nvidia-smi based cleanup could go here if needed in future
