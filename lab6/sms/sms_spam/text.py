from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-zA-Z']+")


def tokenize(text: str) -> list[str]:
    # Lowercase first, then extract word-like tokens with a simple regex.
    return TOKEN_RE.findall(text.lower())
