"""Pull fine-tuned GPT-SoVITS weights from HF Hub into the voice weights root.

Run on a new machine, or to pick up a specific revision of a voice. Requires
`hf auth login` with access to the (private) cnojima/gptsovits-<voice> repos.
Moved from the `voice` project's scripts/fetch_weights.py (commit 650dd4a).

Usage:
    python scripts/fetch_weights.py                  # all voices, latest
    python scripts/fetch_weights.py raidenshogun      # one voice, latest
    python scripts/fetch_weights.py raidenshogun@<commit_or_tag>
"""
import os
import sys

from huggingface_hub import snapshot_download

from voice_services import env
from voice_services.tts.voices import default_weights_root

NAMESPACE = "cnojima"
ALL_VOICES = ["raidenshogun", "ayaka", "ganyu", "hutao", "yaemiko", "firefly", "acheron", "kafka", "juliet", "rosamund"]


def main() -> None:
    env.load_env()
    weights_root = default_weights_root()
    specs = sys.argv[1:] or ALL_VOICES
    for spec in specs:
        voice, _, revision = spec.partition("@")
        repo_id = f"{NAMESPACE}/gptsovits-{voice}"
        local_dir = os.path.join(weights_root, voice)
        print(f"fetching {voice} ({revision or 'main'}) <- {repo_id} ...")
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision or None,
            local_dir=local_dir,
        )
        print(f"  done -> {local_dir}")


if __name__ == "__main__":
    main()
