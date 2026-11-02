"""Respell words before they reach GPT-SoVITS, whose text front end mispronounces some of them.

A lexicon is a JSON object per language, `<dir>/<lang>.json`, mapping a word to the spelling that
makes the voice say it right (`"API": "ay pee eye"`). Matching is whole-word and case-insensitive;
longer entries win. Only the text sent for synthesis changes. The directory is `pronunciations/`
beside this repo, or VOICE_TTS_PRONUNCIATIONS. Edit a file and restart the server, or change it
live through `PATCH /pronunciations/<lang>` (see `update`).

    python -m voice_services.tts.pronunciations en "Our API uses AI."   # show the respelling
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import threading
from pathlib import Path

from voice_services import env

_WORD = r"(?<![\w'])(?:{})(?![\w'])"
_LANG = re.compile(r"[a-z]{2,3}")
_write_lock = threading.Lock()


class Lexicon:
    def __init__(self, entries: dict[str, str]):
        self.entries = dict(entries)  # as written in the file; _entries below is the casefolded lookup
        self._entries = {k.casefold(): v for k, v in entries.items() if k}
        keys = sorted(self._entries, key=len, reverse=True)
        self._pattern = re.compile(_WORD.format("|".join(map(re.escape, keys))), re.IGNORECASE) if keys else None

    def __len__(self) -> int:
        return len(self._entries)

    def apply(self, text: str) -> str:
        if self._pattern is None:
            return text
        return self._pattern.sub(lambda m: self._entries[m.group(0).casefold()], text)


def default_dir() -> Path:
    return Path(os.environ.get("VOICE_TTS_PRONUNCIATIONS") or env.REPO_ROOT / "pronunciations")


def load(directory: Path | None = None) -> dict[str, Lexicon]:
    """One Lexicon per language file found. A missing directory or file means no respelling; a
    malformed file stops startup so a typo is not silently ignored."""
    directory = directory or default_dir()
    out: dict[str, Lexicon] = {}
    for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not all(isinstance(v, str) for v in data.values()):
                raise ValueError("expected an object of word -> spelling strings")
        except (OSError, ValueError) as e:
            raise ValueError(f"{path}: {e}") from e
        out[path.stem] = Lexicon(data)
    return out


def _read(path: Path) -> dict[str, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        raise ValueError(f"{path}: {e}") from e
    if not isinstance(data, dict) or not all(isinstance(v, str) for v in data.values()):
        raise ValueError(f"{path}: expected an object of word -> spelling strings")
    return data


def update(lang: str, set: dict[str, str] | None = None, remove: list[str] | tuple[str, ...] = (),
           directory: Path | None = None) -> Lexicon:
    """Merge `set` into `<dir>/<lang>.json` and drop the `remove` words, then return the new Lexicon.
    Words match case-insensitively, so setting `api` replaces an existing `API`. The file is replaced
    atomically; a ValueError (bad input or a malformed existing file) leaves it untouched."""
    if not _LANG.fullmatch(lang):
        raise ValueError(f"invalid language {lang!r}")
    set = set or {}
    for word, spelling in set.items():
        if not word.strip() or not isinstance(spelling, str) or not spelling.strip():
            raise ValueError("words and spellings must be non-empty strings")
    directory = directory or default_dir()
    path = directory / f"{lang}.json"
    drop = {w.casefold() for w in (*set, *remove)}
    with _write_lock:
        entries = {k: v for k, v in _read(path).items() if k.casefold() not in drop}
        entries.update(set)
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory, suffix=".tmp", delete=False) as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(f.name, path)
    return Lexicon(entries)


def main(argv: list[str] | None = None) -> None:
    env.load_env()
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        sys.exit("usage: python -m voice_services.tts.pronunciations <lang> <text>")
    lang, text = args
    print(load().get(lang, Lexicon({})).apply(text))


if __name__ == "__main__":
    main()
