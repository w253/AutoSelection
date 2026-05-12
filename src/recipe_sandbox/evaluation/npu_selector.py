"""NPU (Huawei Ascend) resource monitor and device selector.

Detects available Ascend NPUs via `npu-smi info` and selects
idle devices for training/evaluation workloads.

Mirrors the API of `gpu_selector.py` so the search loop can
transparently switch between GPU and NPU backends.

npu-smi info output example:
+---------------------------+---------------+------+
| NPU   Name    Health      | AICore(%)     | Mem  |
+===========================+===============+======+
| 0     910B3   OK          | 0             | 1234 / 65536 |
| ...                                               |
+---------------------------+---------------+------+

Also supports:
  - npu-smi info -t memory -i <id>   → detailed HBM per chip
  - npu-smi info -t proc-mem -i <id> → per-process memory
  - torch_npu.npu.device_count()     → PyTorch device count
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
#  NPU Info
# -----------------------------------------------------------------------

@dataclass
class NPUInfo:
    index: int
    name: str
    health: str
    aicore_usage_pct: float      # AICore utilization %
    hbm_used_mb: int             # HBM used (MB)
    hbm_total_mb: int            # HBM total (MB)
    hbm_free_mb: int             # HBM free (MB)

    @property
    def usage_ratio(self) -> float:
        if self.hbm_total_mb == 0:
            return 1.0
        return self.hbm_used_mb / self.hbm_total_mb


# -----------------------------------------------------------------------
#  Query Methods
# -----------------------------------------------------------------------

def query_npu_info() -> List[NPUInfo]:
    """Query npu-smi for NPU device information.

    Parses the table output of `npu-smi info` to extract:
      - Device index, name, health
      - AICore utilization %
      - HBM memory usage (used / total)
    """
    try:
        result = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            logger.warning("npu-smi info failed: %s", result.stderr.strip())
            return []

        parsed = _parse_npu_smi_output(result.stdout)
        if parsed and _parsed_indices_look_valid(parsed):
            return parsed
        if parsed:
            logger.warning("Parsed NPU indices look invalid. Falling back to torch_npu device enumeration.")
        return _query_via_torch_npu()

    except FileNotFoundError:
        logger.warning("npu-smi not found. Trying torch_npu fallback...")
        return _query_via_torch_npu()
    except Exception as exc:
        logger.warning("Failed to query NPU info: %s", exc)
        return []


def _parse_npu_smi_output(output: str) -> List[NPUInfo]:
    """Parse npu-smi info output.

    The output format varies slightly across driver versions.
    We look for lines containing device info with patterns like:
      - NPU index, chip name, health status
      - AICore percentage
      - HBM usage: used / total (MB)
    """
    npus: List[NPUInfo] = []
    lines = output.strip().splitlines()

    i = 0
    while i + 1 < len(lines):
        first = lines[i].strip()
        second = lines[i + 1].strip()
        if not (first.startswith("|") and second.startswith("|")):
            i += 1
            continue

        first_cols = [part.strip() for part in first.strip("|").split("|")]
        second_cols = [part.strip() for part in second.strip("|").split("|")]
        if len(first_cols) < 3 or len(second_cols) < 3:
            i += 1
            continue

        first_tokens = first_cols[0].split()
        second_tokens = second_cols[0].split()
        if len(first_tokens) < 2 or len(second_tokens) < 2:
            i += 1
            continue

        card_idx, name = first_tokens[0], first_tokens[1]
        health = first_cols[1].split()[0] if first_cols[1] else ""
        if not card_idx.isdigit() or not second_tokens[1].isdigit() or health not in {"OK", "Warning", "Fault"}:
            i += 1
            continue

        # The second row carries the runtime-visible chip/phy id used by torch_npu.
        phy_id = int(second_tokens[1])

        metrics_numbers = [int(n) for n in re.findall(r"\d+", second_cols[2])]
        if len(metrics_numbers) < 3:
            i += 1
            continue
        aicore_pct = float(metrics_numbers[0])
        hbm_used = metrics_numbers[-2]
        hbm_total = metrics_numbers[-1]
        hbm_free = max(0, hbm_total - hbm_used)

        npus.append(NPUInfo(
            index=phy_id,
            name=name,
            health=health,
            aicore_usage_pct=aicore_pct,
            hbm_used_mb=hbm_used,
            hbm_total_mb=hbm_total,
            hbm_free_mb=hbm_free,
        ))
        i += 2

    if npus:
        logger.info("Detected %d NPU(s) via npu-smi", len(npus))
    else:
        # Fallback: try simpler parsing for different output formats
        npus = _parse_npu_smi_simple(output)

    return npus


def _parsed_indices_look_valid(npus: List[NPUInfo]) -> bool:
    indices = [n.index for n in npus]
    if not indices or len(set(indices)) != len(indices):
        return False
    try:
        import torch
        import torch_npu  # noqa: F401

        count = torch.npu.device_count()
        if count > 0 and any(idx < 0 or idx >= count for idx in indices):
            return False
    except Exception:
        pass
    return True


def _parse_npu_smi_simple(output: str) -> List[NPUInfo]:
    """Fallback parser for different npu-smi output formats.

    Some versions output a simpler format. Try to extract any
    device info we can find.
    """
    npus = []
    # Look for lines with "NPU" and a device number
    for line in output.splitlines():
        # Try to find "NPU ID: X" or similar
        id_match = re.search(r'(?:NPU|Device)\s*(?:ID)?[\s:]*(\d+)', line, re.IGNORECASE)
        if id_match:
            idx = int(id_match.group(1))
            # Try to find memory info
            mem_match = re.search(r'(\d+)\s*/\s*(\d+)\s*(?:MB)?', line)
            hbm_used = int(mem_match.group(1)) if mem_match else 0
            hbm_total = int(mem_match.group(2)) if mem_match else 65536
            # Try to find utilization
            util_match = re.search(r'(?:AICore|Util)[^\d]*(\d+)\s*%?', line, re.IGNORECASE)
            aicore = float(util_match.group(1)) if util_match else 0.0

            npus.append(NPUInfo(
                index=idx,
                name="Ascend",
                health="OK",
                aicore_usage_pct=aicore,
                hbm_used_mb=hbm_used,
                hbm_total_mb=hbm_total,
                hbm_free_mb=max(0, hbm_total - hbm_used),
            ))
    return npus


def _query_via_torch_npu() -> List[NPUInfo]:
    """Fallback: use torch_npu to detect devices.

    Less detailed than npu-smi but works when npu-smi isn't in PATH.
    """
    try:
        import torch
        import torch_npu

        count = torch.npu.device_count()
        if count == 0:
            return []

        npus = []
        for i in range(count):
            props = torch.npu.get_device_properties(i)
            name = getattr(props, "name", "Ascend")
            total_mem = getattr(props, "total_memory", 0)
            total_mb = total_mem // (1024 * 1024) if total_mem else 65536

            # Try to get current memory usage
            try:
                torch.npu.set_device(i)
                used_mb = torch.npu.memory_allocated(i) // (1024 * 1024)
            except Exception:
                used_mb = 0

            npus.append(NPUInfo(
                index=i,
                name=str(name),
                health="OK",
                aicore_usage_pct=0.0,  # Not available via torch_npu
                hbm_used_mb=used_mb,
                hbm_total_mb=total_mb,
                hbm_free_mb=max(0, total_mb - used_mb),
            ))

        logger.info("Detected %d NPU(s) via torch_npu", len(npus))
        return npus

    except ImportError:
        logger.warning("torch_npu not available. No NPU info.")
        return []
    except Exception as exc:
        logger.warning("torch_npu query failed: %s", exc)
        return []


# -----------------------------------------------------------------------
#  Selector (mirrors gpu_selector API)
# -----------------------------------------------------------------------

def select_idle_npus(
    *,
    min_free_mb: int = 8000,
    max_aicore_pct: float = 30.0,
    max_workers: Optional[int] = None,
) -> List[str]:
    """Select NPUs that are mostly idle.

    Returns a list of device strings, e.g. ["npu:0", "npu:2"].
    Falls back to ["npu:0"] if no idle devices found.
    """
    npus = query_npu_info()
    if not npus:
        logger.info("No NPUs detected.")
        return []

    idle = [
        n for n in npus
        if n.hbm_free_mb >= min_free_mb
        and n.aicore_usage_pct <= max_aicore_pct
        and n.health == "OK"
    ]

    is_fallback = False
    if not idle:
        # Fallback: ignore utilization, sort all NPUs by free memory that satisfy min_free_mb
        healthy = [n for n in npus if n.health == "OK"]
        if not healthy:
            logger.warning("No healthy NPUs found!")
            return []
            
        target_npus = [n for n in healthy if n.hbm_free_mb >= min_free_mb]
        if not target_npus:
            # Absolute fallback if even min_free_mb isn't met: pick the one with most free memory
            target_npus = [max(healthy, key=lambda n: n.hbm_free_mb)]
        is_fallback = True
        idle = target_npus

    # Sort by free HBM descending
    idle.sort(key=lambda n: n.hbm_free_mb, reverse=True)
    if max_workers is not None:
        idle = idle[:max_workers]

    devices = [f"npu:{n.index}" for n in idle]
    
    if is_fallback:
        logger.warning(
            "No strictly idle NPUs found (util <= %.0f%%). Fallback selected %d NPU(s): %s",
            max_aicore_pct, len(devices),
            ", ".join(f"{d} ({n.hbm_free_mb}MB free, {n.aicore_usage_pct}%% AICore)" for d, n in zip(devices, idle)),
        )
    else:
        logger.info(
            "Selected %d idle NPU(s): %s",
            len(devices),
            ", ".join(f"{d} ({n.hbm_free_mb}MB free)" for d, n in zip(devices, idle)),
        )
    return devices


# -----------------------------------------------------------------------
#  Unified selector (auto-detect GPU vs NPU)
# -----------------------------------------------------------------------

def select_idle_devices(
    *,
    min_free_mb: int = 8000,
    max_util_pct: float = 30.0,
    max_workers: Optional[int] = None,
    prefer: str = "auto",
) -> List[str]:
    """Auto-detect and select idle accelerators (GPU or NPU).

    Args:
        prefer: 'gpu', 'npu', or 'auto' (try GPU first, then NPU)

    Returns device strings like ['cuda:0', 'cuda:1'] or ['npu:0', 'npu:2'].
    """
    from recipe_sandbox.evaluation.gpu_selector import select_idle_gpus

    if prefer == "gpu":
        return select_idle_gpus(min_free_mb=min_free_mb, max_util_pct=max_util_pct, max_workers=max_workers)
    elif prefer == "npu":
        return select_idle_npus(min_free_mb=min_free_mb, max_aicore_pct=max_util_pct, max_workers=max_workers)
    else:
        # Auto: try GPU first, then NPU
        gpus = select_idle_gpus(min_free_mb=min_free_mb, max_util_pct=max_util_pct, max_workers=max_workers)
        if gpus and gpus != ["cpu"]:
            return gpus
        npus = select_idle_npus(min_free_mb=min_free_mb, max_aicore_pct=max_util_pct, max_workers=max_workers)
        if npus:
            return npus
        return ["cpu"]


# -----------------------------------------------------------------------
#  CLI: python -m recipe_sandbox.evaluation.npu_selector
# -----------------------------------------------------------------------

def main():
    """Print NPU status (similar to npu-smi info but cleaner)."""
    import json

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=" * 60)
    print("  NPU Device Monitor")
    print("=" * 60)

    npus = query_npu_info()
    if not npus:
        print("\nNo NPUs detected. Trying torch_npu fallback...")
        npus = _query_via_torch_npu()

    if not npus:
        print("No NPUs found on this system.")
        return

    print(f"\nDetected {len(npus)} NPU(s):\n")
    print(f"{'ID':>4}  {'Name':<12}  {'Health':<8}  {'AICore%':>8}  {'HBM Used':>10}  {'HBM Total':>10}  {'HBM Free':>10}  {'Usage%':>7}")
    print("-" * 85)
    for n in npus:
        print(
            f"{n.index:4d}  {n.name:<12}  {n.health:<8}  {n.aicore_usage_pct:7.1f}%  "
            f"{n.hbm_used_mb:8d}MB  {n.hbm_total_mb:8d}MB  {n.hbm_free_mb:8d}MB  "
            f"{n.usage_ratio*100:6.1f}%"
        )

    print()
    idle = select_idle_npus(min_free_mb=8000, max_aicore_pct=30.0)
    print(f"Idle devices (>8GB free, <30% AICore): {idle or 'None'}")


if __name__ == "__main__":
    main()
