"""Phase 1 prep — wrapper around `synapseclient` so the user only does login once.

Usage:
    pip install synapseclient
    python scripts/synapse_login.py        # prompts for username + auth token
    python scripts/synapse_login.py --check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="Just verify existing credentials.")
    args = ap.parse_args()

    try:
        import synapseclient
    except ImportError:
        print("[!] synapseclient not installed. Run: pip install synapseclient")
        return 1

    syn = synapseclient.Synapse(silent=True)
    if args.check:
        try:
            profile = syn.getUserProfile()
            print(f"[ok] logged in as {profile.userName} ({profile.ownerId})")
            return 0
        except Exception as e:  # noqa: BLE001
            print(f"[!] not authenticated: {e}")
            return 2

    print("Visit https://www.synapse.org/#!PersonalAccessTokens: to mint an auth token.")
    user = input("Synapse username: ").strip()
    token = input("Personal access token (paste; will not echo): ").strip()
    if not user or not token:
        print("[!] empty input; aborting")
        return 3

    try:
        syn.login(email=user, authToken=token, rememberMe=True)
        cache = Path.home() / ".synapseConfig"
        print(f"[ok] credentials cached at {cache}")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"[!] login failed: {e}")
        return 4


if __name__ == "__main__":
    sys.exit(main())
