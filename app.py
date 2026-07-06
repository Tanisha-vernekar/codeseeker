#!/usr/bin/env python3
"""Run this one file to launch the codeseeker web app.

    python app.py

It starts a local server and opens the UI in your browser, where you can index
a repository (local folder or a remote ``owner/repo``) and then search, ask
questions, explain the project, and view stats.

Optionally set host/port:

    python app.py --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import os
import sys

# Make sure the package is importable when run directly from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the codeseeker web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    try:
        from codeseeker.webapp import run_server
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("Install the web dependency with: pip install flask", file=sys.stderr)
        return 1

    run_server(host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
