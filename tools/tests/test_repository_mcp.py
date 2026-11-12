import json
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import repository_mcp as mcp


class TransportTests(unittest.TestCase):
    def server(self, directory, script):
        path=Path(directory)/"server.py"; path.write_text(textwrap.dedent(script))
        return [sys.executable,"-u",str(path)]

    def test_serial_requests_notification_and_raw_byte_traces(self):
        with tempfile.TemporaryDirectory() as d:
            cmd=self.server(d, '''
                import sys,json
                for line in sys.stdin:
                    x=json.loads(line)
                    if "id" not in x: continue
                    if x["method"]=="first":
                        print(json.dumps({"jsonrpc":"2.0","method":"notifications/progress","params":{}}),flush=True)
                    print(json.dumps({"jsonrpc":"2.0","id":x["id"],"result":{"method":x["method"]}}),flush=True)
            ''')
            with mcp.JSONLineSession(cmd,d,"trace",timeout_s=1) as session:
                self.assertEqual(session.request("first",{}),{"method":"first"})
                session.notify("ready",{})
                self.assertEqual(session.request("second",{}),{"method":"second"})
                self.assertEqual([r["id"] for r in session.records if "id" in r],[1,2])
            raw=(Path(d)/"trace.stdin").read_bytes()
            self.assertEqual(raw.count(b'"method":"first"'),1)
            self.assertEqual(raw.count(b'"method":"ready"'),1)
            self.assertIn(b'"method":"second"',raw)

    def test_wrong_id_poisons_session(self):
        with tempfile.TemporaryDirectory() as d:
            cmd=self.server(d, '''
                import sys,json
                x=json.loads(sys.stdin.readline())
                print(json.dumps({"jsonrpc":"2.0","id":x["id"]+1,"result":{}}),flush=True)
            ''')
            session=mcp.JSONLineSession(cmd,d,"bad",timeout_s=1)
            with self.assertRaises(mcp.MCPError): session.request("x",{})
            self.assertTrue(session.poisoned)
            with self.assertRaises(mcp.MCPError): session.request("again",{})
            session.close()


class CollectorTests(unittest.TestCase):
    source=b"fn sample() {\n  answer();\n}\n"
    chunk=b"fn sample() {\n  answer();\n"
    mode="ok"

    def server(self, directory, script):
        path=Path(directory)/"server.py"; path.write_text(textwrap.dedent(script))
        return [sys.executable,"-u",str(path)]

    class FakeSession:
        instances=[]
        mode="ok"
        search_count=0
        root_path=None
        search_arguments=[]
        def __init__(self, command, output_dir, stem, timeout_s=30, env=None):
            self.stem=stem; self.records=[]; self.closed=False; self.snapshot=None
            self.__class__.instances.append(self)
        def request(self,method,params):
            self.records.append({"method":method,"name":params.get("name") if isinstance(params,dict) else None})
            if method=="initialize":
                if self.mode=="startup_timeout": raise mcp.MCPError("MCP request deadline exceeded")
                return {"protocolVersion":"2025-11-25"}
            if method=="tools/list": return {"tools":[{"name":"search_code"},{"name":"read_code"}]}
            tool_name=params["name"]; args=params["arguments"]
            if tool_name=="search_code":
                type(self).search_count+=1
                type(self).search_arguments.append(args)
                n=type(self).search_count
                if self.mode=="stale_source": (type(self).root_path/"src/a.rs").write_bytes(CollectorTests.source+b"// changed\n")
                snapshot="snap-changed" if self.mode=="cross_snapshot" and n>1 else "snap-one"
                path="../outside" if self.mode=="path_escape" else "src/a.rs"
                self.snapshot=snapshot
                hit={"chunk_id":n,"path":path,"language":"rust","symbol":"sample","start_line":1,"end_line":2,"content_hash":"hash"}
                return {"structuredContent":{"format_version":1,"snapshot_id":snapshot,"source":"snapshot","strategy":args["strategy"],"candidate_k":args["candidate_k"],"hits":[hit],"embedding_ms":0.1,"retrieval_ms":0.2}}
            if tool_name=="read_code":
                hit={"snapshot_id":self.snapshot,"chunk_id":args["chunk_id"],"path":"src/a.rs","start_line":1,"end_line":2}
                text=CollectorTests.chunk.decode()
                nxt=0 if self.mode=="bad_pagination" else None
                return {"structuredContent":{"format_version":1,"source":"snapshot",**hit,"offset":0,"next_offset":nxt,"total_bytes":len(CollectorTests.chunk),"text":text}}
            raise AssertionError(method)
        def notify(self,*_): return None
        def close(self): self.closed=True

    def collect(self, mode="ok", query_count=2, strategies=("lexical","symbol")):
        import repository_eval as ev
        with tempfile.TemporaryDirectory() as d:
            base=Path(d); root=base/"repo"; (root/"src").mkdir(parents=True); (root/"src/a.rs").write_bytes(self.source)
            queries=[{"id":f"q{i}","repository":"r","category":"unit","text":f"find sample {i}","filter":{},"anchors":[{"id":f"a{i}","path":"src/a.rs","start_line":1,"end_line":2}]} for i in range(query_count)]
            spec={"format_version":1,"name":"fixture","split":"development","repositories":[{"id":"r","revision":"fixture"}],"queries":queries}
            plan=ev.freeze(spec,root)
            binary=base/"cubeql"; binary.write_text("fake"); binary.chmod(0o755)
            server=base/"cubeql-mcp"; server.write_text("fake"); server.chmod(0o755)
            original=CollectorTests.FakeSession
            original.instances=[]; original.search_count=0; original.mode=mode
            original.search_arguments=[]; original.root_path=root
            def execute(command, output, stem, timeout_s):
                if mode=="index_failure": return {"status":"timeout","error":"index deadline","returncode":None,"elapsed_ms":1,"stdout":stem+".stdout","stderr":stem+".stderr","command":command}
                out=Path(output); snap=Path(command[command.index("--output")+1]); snap.write_bytes(b"snapshot")
                stdout=stem+".stdout"; (out/stdout).write_text(json.dumps({"format_version":1,"snapshot":{"root":str(root.resolve()),"corpus_fingerprint":"f","chunks":1}}))
                (out/(stem+".stderr")).write_bytes(b"")
                return {"status":"ok","returncode":0,"elapsed_ms":1,"stdout":stdout,"stderr":stem+".stderr","command":command}
            with patch.object(mcp.rc,"execute",execute), patch.object(mcp,"JSONLineSession",original):
                result=mcp.collect_mcp(plan,{"r":str(root)},binary,server,base/"out",strategies=strategies,candidate_k=20,top_k=10,timeout_s=1)
            return result,original.instances

    def test_one_persistent_session_reused_across_queries_and_strategies(self):
        result,sessions=self.collect()
        self.assertEqual(len(sessions),1,result["manifest"]["traces"][-1])
        self.assertTrue(sessions[0].closed)
        self.assertTrue(all(x["status"]=="ok" for run in result["runs"] for x in run["results"]),result["runs"])
        self.assertTrue(all("path_prefix" not in args for args in CollectorTests.FakeSession.search_arguments))

    def test_path_escape_is_rejected_before_read_code(self):
        result,sessions=self.collect("path_escape",query_count=1,strategies=("lexical",))
        self.assertEqual(result["runs"][0]["results"][0]["status"],"failed")
        self.assertFalse(any(r["name"]=="read_code" for r in sessions[0].records))

    def test_cross_snapshot_and_bad_pagination_fail_queries(self):
        result,_=self.collect("cross_snapshot",query_count=2,strategies=("lexical",))
        self.assertEqual([x["status"] for x in result["runs"][0]["results"]],["ok","failed"],result["runs"])
        result,_=self.collect("bad_pagination",query_count=1,strategies=("lexical",))
        self.assertEqual(result["runs"][0]["results"][0]["status"],"failed")

    def test_startup_timeout_preserves_timeout_denominator(self):
        result,_=self.collect("startup_timeout",query_count=2,strategies=("lexical",))
        self.assertEqual([x["status"] for x in result["runs"][0]["results"]],["timeout","timeout"])

    def test_index_failure_preserves_timeout_denominator(self):
        result,sessions=self.collect("index_failure",query_count=2,strategies=("lexical",))
        self.assertEqual([x["status"] for x in result["runs"][0]["results"]],["timeout","timeout"])
        self.assertEqual(sessions,[])

    def test_source_change_invalidates_successful_collection(self):
        with self.assertRaises(mcp.MCPError): self.collect("stale_source",query_count=1,strategies=("lexical",))

    def test_timeout_poisons_and_is_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            cmd=self.server(d, '''
                import sys,time
                sys.stdin.readline(); time.sleep(3)
            ''')
            session=mcp.JSONLineSession(cmd,d,"slow",timeout_s=.15)
            start=time.monotonic()
            with self.assertRaises(mcp.MCPError): session.request("x",{})
            self.assertLess(time.monotonic()-start,1)
            self.assertTrue(session.poisoned)
            session.close()

    def test_malformed_frame_poisons(self):
        with tempfile.TemporaryDirectory() as d:
            cmd=self.server(d, '''
                import sys
                sys.stdin.readline(); sys.stdout.write("{broken}\\n"); sys.stdout.flush()
            ''')
            session=mcp.JSONLineSession(cmd,d,"malformed",timeout_s=1)
            with self.assertRaises(mcp.MCPError): session.request("x",{})
            self.assertTrue(session.poisoned); session.close()

    def test_invalid_constructor_does_not_launch_or_create_output(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(mcp.MCPError): mcp.JSONLineSession(["echo","x"],d,"x",timeout_s=float("inf"))
            self.assertEqual(list(Path(d).iterdir()),[])

    def test_outgoing_frame_and_nonfinite_params_rejected_before_write(self):
        with tempfile.TemporaryDirectory() as d:
            cmd=self.server(d, '''
                import sys
                sys.stdin.read()
            ''')
            session=mcp.JSONLineSession(cmd,d,"bounded",timeout_s=1)
            with self.assertRaises(mcp.MCPError): session.notify("x",{"v":float("nan")})
            with self.assertRaises(mcp.MCPError): session.notify("x",{"v":"a"*(mcp.FRAME_LIMIT+1)})
            self.assertEqual((Path(d)/"bounded.stdin").read_bytes(),b"")
            session.close()

    def test_duplicate_keys_nonfinite_and_boolean_ids_rejected(self):
        for response in (
            '{"jsonrpc":"2.0","id":1,"result":{"x":1,"x":2}}\n',
            '{"jsonrpc":"2.0","id":1,"result":{"x":NaN}}\n',
            '{"jsonrpc":"2.0","id":true,"result":{}}\n',
        ):
            with self.subTest(response=response), tempfile.TemporaryDirectory() as d:
                escaped=response.replace("\\","\\\\").replace('"','\\"').replace("\n","\\n")
                cmd=self.server(d, f'''\n                    import sys\n                    sys.stdin.readline()\n                    sys.stdout.write("{escaped}")\n                    sys.stdout.flush()\n                ''')
                session=mcp.JSONLineSession(cmd,d,"strict",timeout_s=1)
                with self.assertRaises(mcp.MCPError): session.request("x",{})
                self.assertTrue(session.poisoned); session.close()


if __name__ == "__main__": unittest.main()
