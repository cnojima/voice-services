"""Test-wide setup."""
import os

# The services load a `.env` at startup (voice_services/env.py). A developer's real
# one holds per-machine paths; tests must neither read it nor be changed by it.
os.environ.setdefault("VOICE_ENV_FILE", "")
