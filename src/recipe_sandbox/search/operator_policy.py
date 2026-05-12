from __future__ import annotations

from typing import Iterable, Sequence, Tuple


OFFICIAL_OPERATOR_ORDER: Tuple[str, ...] = (
    "truncate_samples",
    "mona_filter",
    "ifd_filter",
    "ngram_entropy",
    "action_object_branching",
    "varentropy_filter",
    "semantic_dedup",
    "semdedup",
    "union",
)


def resolve_operator_space(
    registered_operators: Iterable[str],
) -> Tuple[str, ...]:
    """Return the official ordered operator vocabulary.

    Only includes operators from ``OFFICIAL_OPERATOR_ORDER`` that are also
    present in *registered_operators*.  This is the single source of truth
    shared by warmup, proposal, restart, and GP encoding.
    """
    registered = set(registered_operators)
    return tuple(op for op in OFFICIAL_OPERATOR_ORDER if op in registered)


_LHS_EXCLUDED = frozenset({"union", "semdedup"})


def resolve_lhs_operator_space(
    operator_space: Sequence[str],
) -> Tuple[str, ...]:
    """Return the LHS-sampable subset of the operator space (excludes union & semdedup)."""
    space_set = set(operator_space)
    return tuple(
        op for op in OFFICIAL_OPERATOR_ORDER
        if op in space_set and op not in _LHS_EXCLUDED
    )
