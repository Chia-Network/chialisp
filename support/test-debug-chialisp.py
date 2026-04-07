#!/usr/bin/env python3
import re
import socket
import subprocess
import sys
import time


BUILD_AND_RUN = [
    "./target/debug/armtx",
    "-o",
    "mandelbrot.elf",
    "resources/tests/mandelbrot/mandelbrot.clsp",
    "(-200 -200 -100 -100 3)",
]


def run_and_capture(cmd):
    proc = subprocess.run(
        cmd,
        cwd=".",
        text=True,
        capture_output=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def build_armtx():
    code, out, err = run_and_capture(["cargo", "build", "--bin", "armtx"])
    if code != 0:
        raise RuntimeError(
            "cargo build --bin armtx failed\n"
            f"stdout:\n{out}\n"
            f"stderr:\n{err}"
        )


def wait_for_stub(proc, timeout_seconds=20):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("armtx exited before gdb connected")
        try:
            with socket.create_connection(("127.0.0.1", 9001), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError("timed out waiting for gdb stub on port 9001")


def run_gdb():
    gdb_cmd = [
        "gdb-multiarch",
        "-q",
        "-ex",
        "set pagination off",
        "-ex",
        "set confirm off",
        "-ex",
        "set breakpoint pending on",
        "-ex",
        "target remote :9001",
        "-ex",
        "source support/gdb_print_sexp.py",
        "-ex",
        "break escape-steps",
        "-ex",
        "continue",
        "-ex",
        "info line *$pc",
        "-ex",
        "frame",
        "-ex",
        "quit",
    ]
    code, out, err = run_and_capture(gdb_cmd)
    return code, out, err


def verify_line_15(gdb_output):
    line_match = re.search(
        r"Line 15 of \"[^\"]*mandelbrot\.clsp\"",
        gdb_output,
        flags=re.IGNORECASE,
    )
    if line_match:
        return True
    return False


def main():
    build_armtx()

    armtx = subprocess.Popen(
        BUILD_AND_RUN,
        cwd=".",
        text=True,
    )

    try:
        wait_for_stub(armtx)
        gdb_code, gdb_stdout, gdb_stderr = run_gdb()
        sys.stdout.write(gdb_stdout)
        sys.stderr.write(gdb_stderr)

        if gdb_code != 0:
            raise RuntimeError(f"gdb-multiarch failed with exit code {gdb_code}")

        if not verify_line_15(gdb_stdout + "\n" + gdb_stderr):
            raise RuntimeError(
                "did not observe PC mapping to line 15 of mandelbrot.clsp"
            )
    finally:
        armtx.terminate()
        try:
            armtx.wait(timeout=5)
        except subprocess.TimeoutExpired:
            armtx.kill()
            armtx.wait(timeout=5)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
