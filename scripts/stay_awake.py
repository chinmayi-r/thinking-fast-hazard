"""Keep Windows from idle-sleeping while HAZARD sweeps run, the way a media player does: assert
ES_SYSTEM_REQUIRED | ES_CONTINUOUS via SetThreadExecutionState every 30 s. Changes no settings; releases on exit.
Usage:  ./venv/Scripts/python.exe scripts/stay_awake.py <log file to watch> <done marker>
Exits when the marker line appears in the log (or on Ctrl-C), then lets the normal power policy resume.
"""
import ctypes
import sys
import time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
log, marker = sys.argv[1], sys.argv[2]
k32 = ctypes.windll.kernel32
print(f"stay_awake: holding system awake until '{marker}' appears in {log}", flush=True)
try:
    while True:
        k32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        try:
            if marker in open(log, encoding="utf-8", errors="replace").read():
                break
        except FileNotFoundError:
            pass
        time.sleep(30)
finally:
    k32.SetThreadExecutionState(ES_CONTINUOUS)
    print("stay_awake: released", flush=True)
