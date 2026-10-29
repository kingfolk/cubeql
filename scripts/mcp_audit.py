#!/usr/bin/env python3
"""Transparent stdio MCP discovery audit, or a non-model initialize/tools-list probe.

Only discovery responses are saved; search arguments, source results and credentials are
not logged. Seeing tools/list proves client discovery, not model adoption or attention.
The parent runner bounds process lifetime and raw output. No shell or config files used.
"""
import argparse
import json
import os
import subprocess
import sys
import threading

MAX_LINE = 1024 * 1024
MAX_AUDIT = 4 * MAX_LINE


class Discovery:
    def __init__(self, output):
        self.output = output
        self.pending = {}
        self.initialized = False
        self.bytes = 0
        self.lock = threading.Lock()

    def observe(self, data, incoming):
        value = json.loads(data)
        if not isinstance(value, dict):
            raise ValueError("MCP message must be an object")
        with self.lock:
            if incoming:
                if value.get("method") in ("initialize", "tools/list") and "id" in value:
                    self.pending[value["id"]] = value["method"]
                return
            method = self.pending.pop(value.get("id"), None)
            if method == "initialize":
                self.initialized = isinstance(value.get("result"), dict) and "error" not in value
            if method == "tools/list":
                result = value.get("result", {})
                record = {"event": "tools_list_response", "initialized": self.initialized,
                          "tools": result.get("tools"), "next_cursor": result.get("nextCursor"),
                          "error": value.get("error")}
                encoded = (json.dumps(record, ensure_ascii=False) + "\n").encode()
                self.bytes += len(encoded)
                if self.bytes > MAX_AUDIT:
                    raise ValueError("discovery audit exceeded limit")
                self.output.write(encoded)
                self.output.flush()


def line(stream):
    value = stream.readline(MAX_LINE + 1)
    if len(value) > MAX_LINE or (value and not value.endswith(b"\n")):
        raise ValueError("oversized or incomplete MCP message")
    return value


def serve(command, audit, probe):
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as child:
        try:
            if probe:
                def request(number, method, params):
                    data = (json.dumps({"jsonrpc": "2.0", "id": number,
                                        "method": method, "params": params}) + "\n").encode()
                    audit.observe(data, True)
                    child.stdin.write(data)
                    child.stdin.flush()
                    while data := line(child.stdout):
                        audit.observe(data, False)
                        value = json.loads(data)
                        if value.get("id") == number:
                            if "error" in value:
                                raise ValueError("MCP probe request failed")
                            return value["result"]
                    raise ValueError("MCP probe ended before response")
                request(1, "initialize", {"protocolVersion": "2025-11-25", "capabilities": {},
                    "clientInfo": {"name": "cubeql-offline-preflight", "version": "2"}})
                child.stdin.write(b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
                child.stdin.flush()
                request(2, "tools/list", {})
                child.stdin.close()
                child.wait(timeout=10)
            else:
                errors = []
                def forward_input():
                    try:
                        while data := line(sys.stdin.buffer):
                            audit.observe(data, True)
                            child.stdin.write(data)
                            child.stdin.flush()
                    except (ValueError, OSError) as error:
                        errors.append(error)
                        child.terminate()
                    finally:
                        child.stdin.close()
                threading.Thread(target=forward_input, daemon=True).start()
                while data := line(child.stdout):
                    audit.observe(data, False)
                    sys.stdout.buffer.write(data)
                    sys.stdout.buffer.flush()
                child.wait(timeout=10)
                if errors:
                    raise ValueError("MCP input forwarding failed")
            if child.returncode:
                raise ValueError("MCP child exited unsuccessfully")
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("MCP command required after --")
    descriptor = os.open(args.audit, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        serve(command, Discovery(output), args.probe)


if __name__ == "__main__":
    main()
