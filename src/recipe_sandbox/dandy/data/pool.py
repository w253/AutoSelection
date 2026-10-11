"""固定图文池与原执行器样本视图；物化始终使用原记录和配方输出顺序。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path

from recipe_sandbox.schema.io import sample_from_dict
from .records import select_mixed_records


@dataclass
class FixedPool:
    """保存原图文记录和轻量 canonical 视图；不在原 schema 添加图文字段。"""

    records: dict
    samples: list

    @classmethod
    def from_config(cls, config):
        """按已有混合采样语义建立固定池，要求稳定唯一 ID。"""
        chosen = select_mixed_records(
            config["datasource"], total_sample=config["total_sample"],
            sample_type=config["sample_type"], seed=config["seed"],
        )
        records, samples = {}, []
        for item in chosen:
            record = deepcopy(item.record)
            sample_id = record["id"]
            if not isinstance(sample_id, str) or not sample_id.strip() or sample_id != sample_id.strip():
                raise ValueError("固定池需要非空、无首尾空白的字符串 ID")
            if sample_id in records:
                raise ValueError(f"固定池重复 ID: {sample_id}")
            records[sample_id] = record
            samples.append(sample_from_dict({
                "sample_id": sample_id, "source_name": item.source_path,
                "messages": deepcopy(record["messages"]),
                "target": {"text": record.get("solution"), "structured": {
                    k: deepcopy(record[k]) for k in ("solution", "verifier_type", "choice_options") if k in record
                }},
            }))
        return cls(records, samples)

    def retain(self, sample_ids):
        """按指标有效池同步样本和记录，不补抽、不改变顺序。"""
        wanted = set(sample_ids)
        self.samples = [s for s in self.samples if s.sample_id in wanted]
        self.records = {s.sample_id: self.records[s.sample_id] for s in self.samples}

    def materialize(self, ids, destination):
        """严格按请求 ID 写原图文 JSONL；未知或重复 ID 在写入前失败。"""
        ids = list(ids)
        if len(ids) != len(set(ids)):
            raise ValueError("候选 ID 重复")
        records = [self.records[sample_id] for sample_id in ids]
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        return path
