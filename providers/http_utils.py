from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def post_json(url: str, payload: dict, headers: dict[str, str], timeout: int = 60) -> dict:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="ignore").strip()
        except Exception:
            body = ""
        detail = f"HTTP Error {exc.code}: {exc.reason}"
        if body:
            detail += f" | body={body[:1200]}"
        raise RuntimeError(detail) from exc
    except URLError as exc:
        raise RuntimeError(f"URL Error: {exc.reason}") from exc
