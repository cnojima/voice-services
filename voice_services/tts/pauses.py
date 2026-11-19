"""Turn dashes, colons and semicolons into commas, which GPT-SoVITS splits fragments on.

The upstream default splitter (cut5) cuts only at a fixed set of punctuation, so an em dash or an
ASCII colon is spoken straight through. Each cut gets a silent gap of `fragment_interval`.
"""
from __future__ import annotations

import re

_COMMA = {"en": ", ", "ja": "、", "fr": ", ", "it": ", "}

# A dash run (with the spaces and stray comma around it), or a colon/semicolon (either width)
# that isn't inside a time/ratio like 10:30 or a scheme like https://.
_DASH = re.compile(r"[,，、]?\s*(?:—|–|―|--)+\s*")
_MARK = re.compile(r"(?<!\d)\s*[:;：；](?!//)(?!\d)\s*|(?<=\d)\s*[;；]\s*|(?<=\d)\s*[:：](?=\D)\s*")


# French typography: guillemets round a quotation, and a space before : ; ? ! (so "a : b" would leave "a , b").
_GUILLEMET = re.compile(r"\s*[«»]\s*")
_SPACE_BEFORE_COMMA = re.compile(r"\s+,")


def apply(text: str, language: str) -> str:
    comma = _COMMA.get(language, ", ")
    if language in ("fr", "it"):
        text = _GUILLEMET.sub(" ", text).strip()
    text = _DASH.sub(comma, text)
    text = _MARK.sub(comma, text)
    return _SPACE_BEFORE_COMMA.sub(",", text) if language in ("fr", "it") else text
