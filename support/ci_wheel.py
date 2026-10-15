"""Install and verify the ABI3 wheel shared by CI tests and publishing."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename


def select_wheel(directory: Path) -> Path:
    supported = set(sys_tags())
    candidates = []
    for wheel in directory.resolve().glob("chialisp-*.whl"):
        _, _, _, tags = parse_wheel_filename(wheel.name)
        if tags & supported:
            if not all(tag.abi == "abi3" for tag in tags):
                raise RuntimeError(f"Expected an ABI3 wheel: {wheel.name}")
            candidates.append(wheel)
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one compatible wheel, found {candidates}")
    return candidates[0]


def verify_wheel(wheel: Path) -> None:
    distribution = importlib.metadata.distribution("chialisp")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    archive = direct_url.get("archive_info", {})
    installed_digest = archive.get("hashes", {}).get("sha256")
    if installed_digest is None:
        installed_digest = archive.get("hash", "").removeprefix("sha256=")
    if direct_url.get("url") != wheel.as_uri() or installed_digest != digest:
        raise RuntimeError("Installed chialisp does not match the release wheel artifact")
    extension = importlib.import_module("chialisp._chialisp")
    installed_package = Path(distribution.locate_file("chialisp")).resolve()
    if not Path(extension.__file__).resolve().is_relative_to(installed_package):
        raise RuntimeError("chialisp extension is imported outside the installed wheel")
    print(f"Verified {wheel.name}: sha256={digest}")
    print(f"Extension: {extension.__file__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["install", "verify"])
    parser.add_argument("wheel_directory", type=Path)
    parser.add_argument("--constraint", type=Path)
    args = parser.parse_args()
    wheel = select_wheel(args.wheel_directory)
    if args.command == "install":
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
             "--force-reinstall", str(wheel)],
            check=True,
        )
        if args.constraint:
            digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
            args.constraint.write_text(f"chialisp @ {wheel.as_uri()}#sha256={digest}\n")
    elif args.constraint:
        parser.error("--constraint requires the install command")
    verify_wheel(wheel)


if __name__ == "__main__":
    main()
