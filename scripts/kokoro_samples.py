"""Synthesize the phrase fixtures with Kokoro voices, for a listening check (not used by the server).

    python scripts/kokoro_samples.py                         # the default candidate voices, all languages
    python scripts/kokoro_samples.py --lang fr --voice ff_siwis
    python scripts/kokoro_samples.py --out samples

Writes <out>/<lang>/<voice>/<nn>.wav (24 kHz mono) and prints the seconds each phrase took, so a
slow phrase shows up beside a mispronounced one. `*.wav` is git-ignored.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "phrases.json"
LANG_CODE = {"en": "a", "ja": "j", "fr": "f", "it": "i"}
CANDIDATES = {
    "fr": ["ff_siwis"],
    "it": ["if_sara", "im_nicola"],
    "en": ["af_heart", "af_bella", "am_michael"],
    "ja": ["jf_alpha", "jf_nezumi", "jm_kumo"],
}
SAMPLE_RATE = 24000


def use_system_espeak() -> None:
    """The espeak-ng that pip bundles (espeakng-loader) has its build machine's data path baked in and crashes on
    macOS. VOICE_ESPEAK_LIBRARY and VOICE_ESPEAK_DATA (e.g. Homebrew's lib/libespeak-ng.dylib and
    share/espeak-ng-data) point misaki at an installed espeak-ng instead."""
    import os

    lib, data = os.environ.get("VOICE_ESPEAK_LIBRARY"), os.environ.get("VOICE_ESPEAK_DATA")
    if not (lib or data):
        return
    import misaki.espeak  # noqa: F401 -- sets the bundled paths on import; override after it
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    if lib:
        EspeakWrapper.set_library(lib)
    if data:
        EspeakWrapper.set_data_path(data)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lang", choices=sorted(LANG_CODE), help="one language (default: all)")
    p.add_argument("--voice", help="one voice (default: the candidates for each language)")
    p.add_argument("--out", default="samples")
    args = p.parse_args()

    use_system_espeak()
    from kokoro import KPipeline  # imported here so --help works without it

    phrases = json.loads(FIXTURES.read_text(encoding="utf-8"))
    for lang in [args.lang] if args.lang else list(LANG_CODE):
        pipeline = KPipeline(lang_code=LANG_CODE[lang])
        for voice in [args.voice] if args.voice else CANDIDATES[lang]:
            out = Path(args.out) / lang / voice
            out.mkdir(parents=True, exist_ok=True)
            for n, text in enumerate(phrases[lang], 1):
                start = time.perf_counter()
                audio = [np.asarray(a, dtype=np.float32) for _, _, a in pipeline(text, voice=voice) if a is not None]
                wav = np.concatenate(audio) if audio else np.zeros(0, dtype=np.float32)
                sf.write(out / f"{n:02d}.wav", wav, SAMPLE_RATE)
                print(f"{lang}/{voice}/{n:02d}  {len(wav) / SAMPLE_RATE:5.1f}s audio in {time.perf_counter() - start:4.1f}s  {text}")


if __name__ == "__main__":
    main()
