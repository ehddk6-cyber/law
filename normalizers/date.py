from __future__ import annotations

import re
from datetime import date


DATE_RE = re.compile(r"^\s*(\d{4})\.(\d{1,2})\.(\d{1,2})\.\s*$")


def parse_korean_date(value: str) -> str | None:
    if not value:
        return None
    match = DATE_RE.match(value)
    if not match:
        return None
    year, month, day = (int(group) for group in match.groups())
    if month == 0 or day == 0:
        return None
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None

