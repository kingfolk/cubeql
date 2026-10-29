#!/usr/bin/env python3
"""OFFLINE protocol fixture, NOT a language model or evidence of retrieval quality."""
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib

if "--version" in sys.argv:
    print("offline-fixture 1")
    sys.exit()
if "--help" in sys.argv:
    print("--strict-config --ignore-user-config --json --ephemeral --output-schema --sandbox")
    sys.exit()

args = sys.argv[1:]
def value(flag):
    return args[args.index(flag) + 1]
def event(value):
    print(json.dumps(value), flush=True)

configs = {}
for i, arg in enumerate(args):
    if arg == "-c":
        key, text = args[i+1].split("=", 1)
        configs[key] = tomllib.loads("value=" + text)["value"]
assert value("--sandbox") == "read-only"
assert "--ignore-user-config" in args and configs["web_search"] == "disabled"
assert configs["features.multi_agent"] is False
workspace = Path(value("--cd"))
assert not (workspace / "evals").exists() and not (workspace / ".codex").exists()
source = (workspace / "lib.rs").read_text()
line = next(i for i, text in enumerate(source.splitlines(), 1) if "pub fn target" in text)
event({"type":"turn.started"})
server = configs.get("mcp_servers.cubeql")
if server:
    child = subprocess.Popen([server["command"], *server["args"]], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    def request(number, method, params):
        child.stdin.write(json.dumps({"jsonrpc":"2.0","id":number,"method":method,"params":params}) + "\n")
        child.stdin.flush()
        while True:
            response = json.loads(child.stdout.readline())
            if response.get("id") == number:
                return response["result"]
    request(1, "initialize", {"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"OFFLINE-FIXTURE","version":"1"}})
    child.stdin.write('{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
    child.stdin.flush()
    strategy = server["args"][server["args"].index("--search-strategy") + 1]
    found = request(2, "tools/call", {"name":"search_code","arguments":{"query":"target","strategy":strategy}})
    assert not found["isError"]
    payload = found["structuredContent"]
    hit = next(h for h in payload["hits"] if h["symbol"] == "target")
    event({"type":"item.started","item":{"id":"search","type":"mcp_tool_call","server":"cubeql","tool":"search_code"}})
    event({"type":"item.completed","item":{"id":"search","type":"mcp_tool_call","server":"cubeql","tool":"search_code","result":found}})
    read = request(3, "tools/call", {"name":"read_code","arguments":{"snapshot_id":payload["snapshot_id"],"chunk_id":hit["chunk_id"]}})
    event({"type":"item.completed","item":{"id":"read","type":"mcp_tool_call","server":"cubeql","tool":"read_code","result":read}})
    child.stdin.close()
    child.wait(timeout=10)
else:
    event({"type":"item.completed","item":{"id":"grep","type":"command_execution","command":"rg target lib.rs","status":"completed"}})
answer = {"locations":[{"path":"lib.rs","symbol":"target","start_line":line,"end_line":line}],
          "explanation":"OFFLINE fixture output, not a model response."}
Path(value("--output-last-message")).write_text(json.dumps(answer))
event({"type":"turn.completed","usage":{"input_tokens":100,"cached_input_tokens":0,"output_tokens":20}})
