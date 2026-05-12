from typing import Iterable


def join_non_empty(parts: Iterable[str], sep: str = "\n\n") -> str:
    return sep.join(part.strip() for part in parts if isinstance(part, str) and part.strip())
