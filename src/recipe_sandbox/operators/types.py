from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class OperatorStats:
    input_samples: int = 0
    output_samples: int = 0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    @property
    def token_delta(self) -> Optional[int]:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.output_tokens - self.input_tokens

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["token_delta"] = self.token_delta
        return payload


@dataclass
class OperatorCost:
    wall_clock_sec: float = 0.0
    cpu_seconds: Optional[float] = None
    gpu_seconds: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OperatorTrace:
    operator_name: str
    operator_type: str
    operator_version: str = "v1"
    fitted: bool = False
    stats: OperatorStats = field(default_factory=OperatorStats)
    cost: OperatorCost = field(default_factory=OperatorCost)
    config: Dict[str, Any] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operator_name": self.operator_name,
            "operator_type": self.operator_type,
            "operator_version": self.operator_version,
            "fitted": self.fitted,
            "stats": self.stats.to_dict(),
            "cost": self.cost.to_dict(),
            "config": self.config,
            "notes": self.notes,
        }
