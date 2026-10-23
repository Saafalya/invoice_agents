"""Check whether this computer can run the agents with Ollama: python check_ollama.py"""
import json
import os
import platform
import subprocess
import urllib.request

OLLAMA = "http://localhost:11434"


def ram_gb():
    system = platform.system()
    try:
        if system == "Windows":
            import ctypes

            class Mem(ctypes.Structure):
                _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong),
                            ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                            ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                            ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong),
                            ("ae", ctypes.c_ulonglong)]
            m = Mem(); m.len = ctypes.sizeof(Mem)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.total / 1024 ** 3
        if system == "Darwin":
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 1024 ** 3
        with open("/proc/meminfo") as f:
            return int(f.readline().split()[1]) / 1024 ** 2
    except Exception:
        return None


def ollama_models():
    try:
        with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=3) as r:
            return [m["name"] for m in json.load(r).get("models", [])]
    except Exception:
        return None


ram = ram_gb()
print(f"System : {platform.system()} {platform.release()} ({platform.machine()})")
print(f"CPU    : {os.cpu_count()} cores")
print(f"RAM    : {ram:.1f} GB" if ram else "RAM    : unknown")

if ram is None:
    pick = "llama3.2"
elif ram >= 12:
    pick = "llama3.2"
elif ram >= 7:
    pick = "qwen2.5:3b"
else:
    pick = "qwen2.5:1.5b"
print(f"Advice : use '{pick}'" + (" (close other apps while it runs)" if ram and ram < 12 else ""))

models = ollama_models()
if models is None:
    print("Ollama : NOT running. Start the Ollama app (or run 'ollama serve'), then re-run this check.")
elif not models:
    print(f"Ollama : running, but no models yet. Run: ollama pull {pick}")
else:
    print("Ollama : running. Models: " + ", ".join(models))
    have = next((m for m in models if m.split(":")[0] == pick.split(":")[0]), models[0])
    print(f"Next   : python run_cli.py --ollama {have} --n 10 --verbose")
