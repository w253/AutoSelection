"""为benchmark提供固定总量的分层采样，不负责SAE划分或读写图片。

调用方先过滤答案并排除固定SAE记录，再传入分层字段。按稳定组名轮流分配名额，
小组耗尽后名额自然转给其他组；返回原输入顺序的记录，保证ID及图片引用不变。
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from itertools import islice
from typing import Any, Iterator, Sequence


def balanced_eval_records(records: list[dict[str, Any]], target: int | None,
                          levels: Sequence[Sequence[str]], mode: str = 'sequential',
                          seed: int = 42) -> list[dict[str, Any]]:
    """按分层轮转选出target条eval，target=None保留全部，不修改输入。

    levels每项是一层的分组字段；例如category再l2_category为两层，
    (level,subject)联合作为一层。sequential在叶组保持原顺序；random按seed与ID
    哈希排序，不依赖全局随机状态。容量不足、重复ID或非法参数时报错，避免静默减量。
    """
    ids = [row['id'] for row in records]
    if len(set(ids)) != len(ids):
        raise ValueError('eval候选ID重复')
    if mode not in {'sequential', 'random'}:
        raise ValueError('采样模式必须为sequential或random')
    if target is None:
        return list(records)
    if isinstance(target, bool) or not isinstance(target, int) or target <= 0 or target > len(records):
        raise ValueError(f'eval目标数量非法或容量不足：目标{target}，可用{len(records)}')

    def ordered(rows: list[dict[str, Any]], depth: int) -> Iterator[dict[str, Any]]:
        """以rows和层深depth递归生成均衡记录流，无文件或随机状态副作用。

        depth达到levels长度时输出叶组顺序；否则按下一层分组轮转yield记录，
        自动跳过已经耗尽的分组，供外层截取总预算。
        """
        if depth == len(levels):
            if mode == 'random':
                rows = sorted(rows, key=lambda row: hashlib.sha256(
                    f"{seed}:{row['id']}".encode()).digest())
            yield from rows
            return
        groups = defaultdict(list)
        for row in rows:
            groups[tuple(str(row[key]) for key in levels[depth])].append(row)
        streams = [ordered(groups[key], depth + 1) for key in sorted(groups)]
        while streams:
            remaining = []
            for stream in streams:
                try:
                    yield next(stream)
                    remaining.append(stream)
                except StopIteration:
                    pass
            streams = remaining

    selected = {row['id'] for row in islice(ordered(records, 0), target)}
    return [row for row in records if row['id'] in selected]
