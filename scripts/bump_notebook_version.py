#!/usr/bin/env python3
"""Stamp cell-02-title in the notebook with the current local date/time and timezone.

Run automatically by the pre-commit hook whenever the notebook is staged.
Can also be run manually: python3 scripts/bump_notebook_version.py
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
NOTEBOOK  = REPO_ROOT / "stable_audio_3_demo.ipynb"
TARGET_ID = "cell-02-title"
# Matches any prior stamp regardless of timezone label (UTC, IDT, EST, +0300, …)
VERSION_RE = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2} \S+")

_local = datetime.now().astimezone()
_tz    = _local.strftime("%Z") or _local.strftime("%z")
STAMP  = _local.strftime("%Y-%m-%d %H:%M") + f" {_tz}"


def main() -> None:
    try:
        nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"bump_notebook_version: cannot read notebook: {exc}", file=sys.stderr)
        sys.exit(1)

    for cell in nb["cells"]:
        if cell.get("id") != TARGET_ID:
            continue
        source = cell["source"]
        # source is a list of line-strings in standard .ipynb format
        if isinstance(source, list):
            for i, line in enumerate(source):
                if VERSION_RE.search(line):
                    source[i] = VERSION_RE.sub(STAMP, line)
                    break
            else:
                print("bump_notebook_version: version pattern not found", file=sys.stderr)
                sys.exit(1)
        else:
            new_src, n = VERSION_RE.subn(STAMP, source)
            if not n:
                print("bump_notebook_version: version pattern not found", file=sys.stderr)
                sys.exit(1)
            cell["source"] = new_src
        break
    else:
        print(f"bump_notebook_version: cell '{TARGET_ID}' not found", file=sys.stderr)
        sys.exit(1)

    NOTEBOOK.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(f"bump_notebook_version: stamped {STAMP}")


if __name__ == "__main__":
    main()
