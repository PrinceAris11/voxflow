import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SERVER_DIR = ROOT / "services" / "ml-engine"
CLIENT_DIR = ROOT / "apps" / "desktop"


import urllib.request


def wait_for_server(host="127.0.0.1", port=8000, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=1) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.25)
    return False


def start_server():
    if wait_for_server(timeout=0.8):
        return None
    proc = subprocess.Popen(
        [sys.executable, str(SERVER_DIR / "demo_server.py")],
        cwd=str(SERVER_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if not wait_for_server():
        output = proc.stdout.read() if proc.stdout else ""
        raise RuntimeError(f"Server did not start in time. Output:\n{output}")
    return proc


def run_client():
    proc = subprocess.run(
        [sys.executable, str(CLIENT_DIR / "demo_client.py")],
        cwd=str(CLIENT_DIR),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Client failed with exit code {proc.returncode}:\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


if __name__ == "__main__":
    server_proc = start_server()
    try:
        print(run_client())
    finally:
        if server_proc and server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()
