#!/usr/bin/env python3
"""
server_debug_wrapper.py

Launches the FastAPI app under Uvicorn with debug log level, captures stdout/stderr
with UTF-8 decoding (ignoring illegal sequences), and optionally sends a single
test request to /chat (if a valid token is provided).

It first checks whether the port is free; if the port is in use, it prints a clear
message and exits. It also checks VELTRAX_API_TOKEN; if not set, it warns and
skips sending the test request to avoid Illegal header errors.

Usage:
    python server_debug_wrapper.py [--port PORT] [--test-after-start]

Options:
    --port PORT           Port to bind Uvicorn (default: 8000)
    --test-after-start    If provided, sends one test request after startup.
"""

import subprocess
import threading
import sys
import time
import os
import asyncio
import socket
import argparse
import httpx
import traceback

def is_port_in_use(host: str, port: int) -> bool:
    """
    Return True if binding to (host, port) fails because address is in use.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True

def run_uvicorn(log_lines, host: str, port: int):
    """
    Launches Uvicorn as a subprocess, capturing stdout and stderr.
    Decodes output as UTF-8 (ignoring errors) to avoid UnicodeDecodeError on Windows.
    Each decoded line is appended to log_lines and printed immediately.
    """
    cmd = [
        sys.executable,
        "-m", "uvicorn",
        "api_server:app",
        "--host", host,
        "--port", str(port),
        "--log-level", "debug"
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except Exception as e:
        print(f"[wrapper] Failed to start Uvicorn subprocess: {e}")
        sys.exit(1)

    def reader(pipe):
        try:
            while True:
                chunk = pipe.readline()
                if not chunk:
                    break
                if isinstance(chunk, bytes):
                    text = chunk.decode("utf-8", errors="ignore")
                else:
                    text = chunk
                log_lines.append(text)
                print(text, end="")
        except Exception:
            tb = traceback.format_exc()
            log_lines.append(tb)
            print(tb, end="")

    threading.Thread(target=reader, args=(proc.stdout,), daemon=True).start()
    threading.Thread(target=reader, args=(proc.stderr,), daemon=True).start()
    return proc

async def send_test_request(host: str, port: int):
    """
    Sends one POST to /chat to test the server.
    Reads VELTRAX_API_TOKEN from environment. If missing or empty, warns and returns.
    Prints status code & body or full exception traceback.
    """
    token = os.getenv("VELTRAX_API_TOKEN", "").strip()
    if not token:
        print("[wrapper] Warning: environment variable VELTRAX_API_TOKEN is not set or empty. Skipping test request.")
        return

    url = f"http://{host}:{port}/chat"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {"prompt": "Supervisor debug: ping", "history": []}

    print(f"[wrapper] Sending test request to {url} ...")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json=payload, headers=headers)
            print(f"[test] Status: {resp.status_code}")
            print(f"[test] Body: {resp.text}")
    except Exception:
        print("[test] Exception during request:")
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Launch Uvicorn with debug logs and optional test request.")
    parser.add_argument("--port", type=int, default=8000, help="Port for Uvicorn (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for Uvicorn (default: 127.0.0.1)")
    parser.add_argument("--test-after-start", action="store_true",
                        help="If set, send one test request after startup (requires VELTRAX_API_TOKEN).")
    parser.add_argument("--startup-wait", type=int, default=5,
                        help="Seconds to wait after starting server before sending test (default: 5)")
    args = parser.parse_args()

    host = args.host
    port = args.port

    # Check port availability
    if is_port_in_use(host, port):
        print(f"[wrapper] Error: Port {port} on host {host} appears to be in use. Stop any existing server or choose a different port.")
        sys.exit(1)

    log_lines = []
    print(f"[wrapper] Starting Uvicorn server on {host}:{port} ...")
    proc = run_uvicorn(log_lines, host, port)

    try:
        # Wait for server startup
        print(f"[wrapper] Waiting {args.startup_wait} seconds for startup...")
        time.sleep(args.startup_wait)

        if args.test_after_start:
            # Send test request
            asyncio.run(send_test_request(host, port))

            # Wait a bit to capture logs
            time.sleep(2)
        else:
            print("[wrapper] Skipping test request (use --test-after-start to enable).")
            # Keep running until user interrupts
            print("[wrapper] Press Ctrl+C to stop server and collect logs.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n[wrapper] KeyboardInterrupt received, shutting down.")
    finally:
        print("[wrapper] Terminating Uvicorn server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("\n=== Collected Uvicorn Logs ===")
        for line in log_lines:
            print(line, end="")

if __name__ == "__main__":
    main()
