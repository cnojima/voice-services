"""Turn dashes, colons and semicolons into commas, which GPT-SoVITS splits fragments on.

The upstream default splitter (cut5) cuts only at a fixed set of punctuation, so an em dash or an
ASCII colon is spoken straight through. Each cut gets a silent gap of `fragment_interval`.
"""
from __future__ import annotations

import re

_COMMA = {"en": ", ", "ja": "、"}

# A dash run (with the spaces and stray comma around it), or a colon/semicolon (either width)
# that isn't inside a time/ratio like 10:30 or a scheme like https://.
_DASH = re.compile(r"[,，、]?\s*(?:—|–|―|--)+\s*")
_MARK = re.compile(r"(?<!\d)\s*[:;：；](?!//)(?!\d)\s*|(?<=\d)\s*[;；]\s*|(?<=\d)\s*[:：](?=\D)\s*")


def apply(text: str, language: str) -> str:
    comma = _COMMA.get(language, ", ")
    text = _DASH.sub(comma, text)
    return _MARK.sub(comma, text)
