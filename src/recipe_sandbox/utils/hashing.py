import hashlib
import json
from typing import Any


def stable_md5(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(payload.encode("utf-8", errors="surrogatepass")).hexdigest()
