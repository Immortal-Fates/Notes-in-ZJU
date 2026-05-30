#!/usr/bin/env python3
"""Google Drive OAuth re-authorization helper.

Run this script, open the printed URL in your browser, complete the Google consent,
and the script will automatically capture the redirect callback and save the token.
"""
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TOKEN_PATH = Path.home() / ".config" / "google-drive-paper-upload" / "token.json"
CLIENT_SECRET = Path.home() / ".config" / "google-drive-paper-upload" / "client_secret.json"


def main():
    if not CLIENT_SECRET.exists():
        print(f"ERROR: client_secret.json not found at {CLIENT_SECRET}", file=sys.stderr)
        print("Expected location: ~/.config/google-drive-paper-upload/client_secret.json", file=sys.stderr)
        return 1

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print("ERROR: google_auth_oauthlib not installed", file=sys.stderr)
        print("Run: pip install --break-system-packages google-auth-oauthlib", file=sys.stderr)
        return 1

    SCOPES = ["https://www.googleapis.com/auth/drive.file"]

    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET), SCOPES)

    print("=" * 60)
    print("Starting Google Drive OAuth flow...")
    print("A browser window should open. If it doesn't, check the URL below:")
    print("=" * 60)
    print()

    try:
        credentials = flow.run_local_server(port=8080, open_browser=True)
    except Exception as e:
        print(f"\nWARNING: Could not auto-open browser: {e}", file=sys.stderr)
        print("Trying manual URL approach...", file=sys.stderr)

        auth_url, _ = flow.authorization_url(
            access_type='offline', include_granted_scopes='true', prompt='consent'
        )
        print("\n" + "=" * 60)
        print("OPEN THIS URL IN YOUR BROWSER:")
        print(auth_url)
        print("=" * 60)
        print("\nAfter authorization, your browser will redirect to http://localhost:8080/")
        print('If you are on a remote machine, set up SSH port forwarding first:')
        print("  ssh -L 8080:localhost:8080 user@host")
        print()

        credentials = flow.run_local_server(port=8080, open_browser=False)

    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(credentials.to_json(), encoding="utf-8")
    print(f"\nToken saved to {TOKEN_PATH}")
    print(f"Expires: {credentials.expiry}")
    print("Done! You can now run the upload script.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
