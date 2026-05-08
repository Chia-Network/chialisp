#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CHIALISP_DEP_RE = re.compile(
    r'^(?P<prefix>\s*chialisp\s*=\s*\{\s*git\s*=\s*"https://github\.com/Chia-Network/chialisp\.git"\s*,\s*rev\s*=\s*")'
    r'(?P<rev>[A-Fa-f0-9]+)'
    r'(?P<suffix>"\s*\}\s*)$'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace chialisp git dependency rev values in matching lines."
        )
    )
    parser.add_argument(
        "new_rev",
        help='New rev value to set (must match "[A-Fa-f0-9]+").',
    )
    parser.add_argument(
        "files",
        nargs="*",
        help=(
            "Files to edit in place. If omitted, recursively scans for Cargo.toml "
            "files from the current directory."
        ),
    )
    return parser.parse_args()


def update_file(path: Path, new_rev: str) -> int:
    try:
        original = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as err:
        print(f"Skipping {path}: {err}", file=sys.stderr)
        return 0

    replacements = 0
    out_lines: list[str] = []

    for line in original.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        ending = line[len(stripped) :]

        match = CHIALISP_DEP_RE.match(stripped)
        if match is None:
            out_lines.append(line)
            continue

        replacements += 1
        out_lines.append(
            f'{match.group("prefix")}{new_rev}{match.group("suffix")}{ending}'
        )

    if replacements > 0:
        path.write_text("".join(out_lines), encoding="utf-8")

    return replacements


def main() -> int:
    args = parse_args()

    if re.fullmatch(r"[A-Fa-f0-9]+", args.new_rev) is None:
        print('Error: new_rev must match "[A-Fa-f0-9]+".', file=sys.stderr)
        return 2

    if args.files:
        targets = [Path(p) for p in args.files]
    else:
        targets = sorted(Path(".").rglob("Cargo.toml"))

    if not targets:
        print("No target files found.", file=sys.stderr)
        return 1

    total_replacements = 0
    touched_files = 0

    for target in targets:
        if not target.is_file():
            print(f"Skipping {target}: not a file", file=sys.stderr)
            continue

        count = update_file(target, args.new_rev)
        if count > 0:
            touched_files += 1
            total_replacements += count
            print(f"Updated {target}: {count} replacement(s)")

    if total_replacements == 0:
        print("No matching lines found.")
        return 1

    print(
        f"Done. Updated {total_replacements} matching line(s) in "
        f"{touched_files} file(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
