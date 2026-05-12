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
    present in *registered_operators*, then appends any extra registered
    operators in registration order.  This is the single source of truth shared
    by warmup, proposal, restart, and GP encoding.
    """
    registered_order = tuple(dict.fromkeys(str(op) for op in registered_operators))
    registered = set(registered_order)
    official = tuple(op for op in OFFICIAL_OPERATOR_ORDER if op in registered)
    extras = tuple(op for op in registered_order if op not in OFFICIAL_OPERATOR_ORDER)
    return official + extras


_LHS_EXCLUDED = frozenset({"union", "semdedup"})


def resolve_lhs_operator_space(
    operator_space: Sequence[str],
) -> Tuple[str, ...]:
    """Return the LHS-sampable subset of the operator space (excludes union & semdedup)."""
    return tuple(op for op in operator_space if op not in _LHS_EXCLUDED)
