"""Text normalization shared by the STT hallucination filter.

Moved from the `voice` project's voice/lang.py (commit 650dd4a); only
`normalize_text` is needed here.
"""
from __future__ import annotations

import re
import unicodedata


def normalize_text(text: str) -> str:
    """Lowercase, drop punctuation, fold accents, collapse whitespace.

    Accents are folded because Whisper is inconsistent about them ("vídeo" vs
    "video"). The kana voicing marks (dakuten/handakuten) are kept: dropping
    them would turn が into か.
    """
    decomposed = unicodedata.normalize("NFKD", text.lower())
    kept = "".join(
        c for c in decomposed
        if unicodedata.category(c) != "Mn" or c in "\u3099\u309a"
    )
    kept = unicodedata.normalize("NFC", kept)
    return re.sub(r"\s+", " ", re.sub(r"[^\w ]", "", kept)).strip()
