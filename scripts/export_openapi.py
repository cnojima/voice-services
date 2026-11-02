"""Write the OpenAPI spec for the pronunciation endpoints to docs/openapi-pronunciations.json.

    python scripts/export_openapi.py          # or --all for every TTS endpoint
    python -m voice_services.tts              # then browse http://127.0.0.1:8772/docs (Swagger UI)

The running server always serves the full spec at /openapi.json and Swagger UI at /docs; this
file is the checked-in copy for the pronunciation feature alone.
"""
import json
import sys
from pathlib import Path

from voice_services import env
from voice_services.tts.server import app

OUT = env.REPO_ROOT / "docs" / "openapi-pronunciations.json"


def refs(node, found: set[str]) -> None:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
            found.add(ref.rsplit("/", 1)[1])
        for v in node.values():
            refs(v, found)
    elif isinstance(node, list):
        for v in node:
            refs(v, found)


def feature_spec(full: dict) -> dict:
    spec = {**full, "paths": {p: v for p, v in full["paths"].items() if p.startswith("/pronunciations/")}}
    spec["info"] = {**full["info"], "title": "voice-services TTS: pronunciations"}
    schemas = full.get("components", {}).get("schemas", {})
    used: set[str] = set()
    todo = [spec["paths"]]
    while todo:
        found: set[str] = set()
        refs(todo.pop(), found)
        todo += [schemas[n] for n in found - used]
        used |= found
    spec["components"] = {**full["components"], "schemas": {n: schemas[n] for n in sorted(used)}}
    return spec


def main() -> None:
    spec = app.openapi()
    if "--all" not in sys.argv[1:]:
        spec = feature_spec(spec)
    OUT.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {Path(OUT).relative_to(env.REPO_ROOT)}: {', '.join(spec['paths'])}")


if __name__ == "__main__":
    main()
