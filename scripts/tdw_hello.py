"""TDW smoke test on the laptop: the build is started by this script from G (A3), not auto-downloaded.

Usage (Git Bash, from G:\iw-thinking-fast):
    ./venv/Scripts/python.exe scripts/tdw_hello.py [--port 1071]

Mirrors tdw/Documentation/lessons/setup/pc.md:25-31 but with check_version=False and launch_build=False,
which is exactly how HAZARD constructs its controller (src/HAZARD/challenge.py:65-74).
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_BUILD = Path(r"G:\iw-thinking-fast\tdw_build\TDW\TDW.exe")  # Linux: .../TDW/TDW.x86_64


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1071)
    ap.add_argument("--build", type=Path, default=DEFAULT_BUILD, help="path to TDW.exe or TDW.x86_64")
    args = ap.parse_args()
    BUILD = args.build
    assert BUILD.is_file(), f"build not found at {BUILD}"

    t0 = time.perf_counter()
    build = subprocess.Popen([str(BUILD), f"-port {args.port}"])
    print(f"started {BUILD.name} pid={build.pid} on port {args.port}")

    from tdw.controller import Controller  # imported after the build starts, like TDW's own docs

    c = Controller(port=args.port, check_version=False, launch_build=False)
    t1 = time.perf_counter()
    print(f"controller connected after {t1 - t0:.2f}s")
    print(f"build version {c._tdw_version}, unity {c._unity_version}, standalone {c._is_standalone}")
    for i in range(3):
        ts = time.perf_counter()
        c.communicate([])  # one physics frame
        print(f"empty communicate #{i}: {(time.perf_counter() - ts) * 1000:.1f} ms")
    print("Hello world!")
    c.communicate({"$type": "terminate"})
    try:
        build.wait(timeout=15)
    except subprocess.TimeoutExpired:
        build.kill()
        print("build did not exit on terminate; killed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
