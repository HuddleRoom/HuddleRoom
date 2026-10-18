import json
import multiprocessing
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tests.live_acceptance.gating_proxy import GatingProxy
from tests.live_acceptance.spend_ledger import BudgetRefused, Ledger

pytestmark = pytest.mark.live
SECRET = "sk-selfcheck-secret-123"
USAGE = {"prompt_tokens": 100, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 30}}
REQ = {"messages": [], "max_tokens": 50}


class FakeUpstream:
    """mode: ok | null_usage | err500 | auth401 | hang | sse | sse_nousage | sse_reject_opts"""

    def __init__(self, mode="ok"):
        self.mode, self.calls, self.auth, self.bodies = mode, 0, [], []
        o = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, status, obj):
                b = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                o.calls += 1
                o.auth.append(self.headers.get("Authorization"))
                o.bodies.append(body)
                m = o.mode
                if m == "ok":
                    return self._json(200, {"choices": [], "usage": USAGE})
                if m == "null_usage":
                    return self._json(200, {"choices": [], "usage": None})
                if m == "err500":
                    return self._json(500, {"error": "boom"})
                if m == "auth401":
                    return self._json(401, {"error": "bad key"})
                if m == "hang":
                    time.sleep(3)
                    return self._json(200, {"usage": USAGE})
                if m == "sse_reject_opts" and "stream_options" in body:
                    return self._json(400, {"error": {"message": "unknown stream_options"}})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                chunks = [{"choices": [{"delta": {"content": "a"}}]}, {"choices": [{"delta": {"content": "b"}}]}]
                if m == "sse" and body.get("stream_options", {}).get("include_usage"):
                    chunks.append({"choices": [], "usage": USAGE})
                for c in chunks:
                    self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/v1"


def post(proxy, label="a", body=None):
    req = urllib.request.Request(proxy.base_url + "/chat/completions", json.dumps(body or REQ).encode(),
                                 {"Content-Type": "application/json", "X-Live-Label": label})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def est_of(body):
    import math
    return math.ceil(len(json.dumps(body)) / 3) + body["max_tokens"]


def _pending(led):
    return Ledger._sum(led._rows())[1]


@pytest.fixture
def env(tmp_path):
    os.environ["SELFCHECK_KEY"] = SECRET
    made = []

    def make(mode="ok", global_cap=10_000, caps=None, url=None, **kw):
        up = FakeUpstream(mode)
        led = Ledger(tmp_path / "l.jsonl", global_cap, caps or {})
        p = GatingProxy(0, tmp_path / "p.jsonl", led, lambda: "s1", upstream_url=url or up.url,
                        upstream_key_env="SELFCHECK_KEY", **kw).start()
        made.append((p, up))
        return led, p, up

    yield make
    for p, up in made:
        p.stop()
        up.srv.shutdown()


def test_usage_recorded_and_key_not_logged(env, tmp_path):
    led, p, up = env()
    assert post(p)[0] == 200
    assert led.totals()["s1"] == {"non_cached": 90, "cached": 30}
    assert up.auth == ["Bearer " + SECRET]
    assert SECRET not in open(tmp_path / "p.jsonl").read() + open(tmp_path / "l.jsonl").read()
    assert p.requests("a")[0]["usage"]["cached"] == 30


def test_cap_refusal_never_calls_upstream(env):
    led, p, up = env(caps={"s1": 1000})
    status, body = post(p, body={"messages": [], "max_tokens": 2000})
    assert status == 429 and json.loads(body)["error"]["reason"] == "budget" and up.calls == 0


def test_upstream_401_records_zero_and_fails_closed(env):
    led, p, up = env("auth401")
    assert post(p)[0] == 401
    assert led.totals()["global"]["non_cached"] == 0 and _pending(led) == {}
    assert p.fatal_error["status"] == 401 and SECRET not in json.dumps(p.fatal_error)
    status, body = post(p)
    assert status == 503 and json.loads(body)["error"]["reason"] == "fatal_upstream_auth"
    assert up.calls == 1 and led.totals()["global"]["non_cached"] == 0


def test_null_usage_records_estimate(env):
    led, p, up = env("null_usage")
    post(p)
    assert led.totals()["global"]["non_cached"] == est_of(REQ)


def test_upstream_500_records_estimate(env):
    led, p, up = env("err500")
    assert post(p)[0] == 500
    assert led.totals()["global"]["non_cached"] == est_of(REQ)


def test_timeout_records_estimate(env):
    led, p, up = env("hang", upstream_timeout=0.5)
    assert post(p)[0] == 502
    assert led.totals()["global"]["non_cached"] == est_of(REQ)


def test_connect_refused_released(env):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    led, p, up = env(url=f"http://127.0.0.1:{port}/v1")
    assert post(p)[0] == 502
    assert led.totals()["global"]["non_cached"] == 0


def test_streaming_with_usage_chunk(env):
    led, p, up = env("sse")
    status, body = post(p, body={**REQ, "stream": True})
    assert status == 200 and body.count(b"data:") == 4 and b"[DONE]" in body
    assert up.bodies[0]["stream"] is True and up.bodies[0]["stream_options"]["include_usage"] is True
    assert led.totals()["s1"] == {"non_cached": 90, "cached": 30}


def test_streaming_without_usage_records_estimate(env):
    led, p, up = env("sse_nousage")
    status, body = post(p, body={**REQ, "stream": True})
    assert status == 200 and body.count(b"data:") == 3
    assert led.totals()["global"]["non_cached"] == est_of({**REQ, "stream": True})


def test_stream_options_rejected_retries_without(env):
    led, p, up = env("sse_reject_opts")
    status, body = post(p, body={**REQ, "stream": True})
    assert status == 200 and up.calls == 2 and "stream_options" not in up.bodies[1]
    assert led.totals()["global"]["non_cached"] == est_of({**REQ, "stream": True})


def test_gate_holds_until_release(env):
    led, p, up = env()
    p.hold("g")
    out = []
    t = threading.Thread(target=lambda: out.append(post(p, "g")))
    t.start()
    assert p.wait_in_flight("g", 5)
    time.sleep(1)
    assert not out and up.calls == 0
    p.release("g")
    t.join(5)
    assert out[0][0] == 200 and up.calls == 1
    assert not p.wait_in_flight("g", 0.2)


def test_star_gate_counts_under_label_and_streaming_gets_zero_bytes(env):
    led, p, up = env("sse")
    p.hold("*")
    s = socket.create_connection(("127.0.0.1", p.port))
    payload = json.dumps({**REQ, "stream": True}).encode()
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nX-Live-Label: lbl\r\nContent-Length: "
              + str(len(payload)).encode() + b"\r\n\r\n" + payload)
    assert p.wait_in_flight("lbl", 5) and p.wait_in_flight("*", 1)
    s.settimeout(0.7)
    with pytest.raises(socket.timeout):
        s.recv(1)
    p.release("*")
    s.settimeout(10)
    buf = b""
    while b"[DONE]" not in buf:
        chunk = s.recv(4096)
        assert chunk
        buf += chunk
    s.close()


def test_gate_timeout_504_records_nothing(env):
    led, p, up = env(gate_timeout=0.3)
    p.hold("g")
    assert post(p, "g")[0] == 504 and up.calls == 0
    assert led.totals()["global"]["non_cached"] == 0


def test_stopped_proxy_refuses_and_missing_key_raises(env, tmp_path):
    led, p, up = env()
    p._stopped = True
    assert post(p)[0] == 503
    p._stopped = False
    os.environ.pop("SELFCHECK_MISSING", None)
    with pytest.raises(RuntimeError):
        GatingProxy(0, tmp_path / "x.jsonl", led, lambda: "s", upstream_url="http://x",
                    upstream_key_env="SELFCHECK_MISSING")


def test_require_bound_refuses_unbounded(env):
    led, p, up = env(require_bound=True)
    assert post(p, body={"messages": []})[0] == 400 and up.calls == 0


def test_negative_estimate_rejected(tmp_path):
    with pytest.raises(ValueError):
        Ledger(tmp_path / "l.jsonl", 100).reserve("s", -1)


def test_record_validates_and_clamps(tmp_path):
    led = Ledger(tmp_path / "l.jsonl", 1000)
    with pytest.raises(ValueError):
        led.record("s", "x", -1, 0)
    led.record("s", "x", 10, 5, 99)  # cached clamped to input
    assert led.totals()["s"] == {"non_cached": 5, "cached": 10}


def test_canned_unexpected_call(tmp_path):
    led = Ledger(tmp_path / "l.jsonl", 100)
    p = GatingProxy(0, tmp_path / "p.jsonl", led, lambda: "s", mode="canned", canned={"ok": {"x": 1}}).start()
    try:
        assert post(p, "ok") == (200, b'{"x": 1}')
        assert post(p, "nope")[0] == 500
    finally:
        p.stop()
    assert [u["label"] for u in p.unexpected_calls()] == ["nope"]


def test_overshoot_recorded_then_refused(env):
    led, p, up = env(global_cap=100)
    small = {"messages": [], "max_tokens": 1}
    assert post(p, body=small)[0] == 200  # estimate ~ 13, actual 90
    assert led.totals()["global"]["non_cached"] == 90
    assert post(p, body=small)[0] == 429


def test_threads_at_cap_bounded_overshoot(env):
    n, cap = 8, 1000
    led, p, up = env(global_cap=cap)
    body = {"messages": [], "max_tokens": 20}
    with ThreadPoolExecutor(n) as ex:
        codes = [c for c, _ in ex.map(lambda _: post(p, body=body), range(40))]
    total = led.totals()["global"]["non_cached"]
    assert 429 in codes and total <= cap + n * (90 - est_of(body)) and total >= cap - 90 - est_of(body)


def _reserve_and_die(path):
    Ledger(path, 1000).reserve("s", 77)


def test_stale_reservation_reaped(tmp_path):
    path = str(tmp_path / "l.jsonl")
    led = Ledger(path, 1000)
    pr = multiprocessing.Process(target=_reserve_and_die, args=(path,))
    pr.start()
    pr.join(30)
    assert led.reap(3600) == 1 and led.reap(3600) == 0
    assert led.totals()["s"]["non_cached"] == 77
    led.reserve("s", 5)  # own live pid, fresh: not reaped
    assert led.reap(3600) == 0 and led.reap(-1) == 1


def test_torn_last_line_skipped_middle_raises(tmp_path):
    path = tmp_path / "l.jsonl"
    led = Ledger(path, 1000)
    led.record("s", "x", 10, 0)
    with open(path, "a") as f:
        f.write('{"k": "usa')
    assert led.totals()["s"]["non_cached"] == 10
    led.record("s", "y", 5, 0)  # truncates torn tail
    assert led.totals()["s"]["non_cached"] == 15
    with open(path, "a") as f:
        f.write('{"k": "usa\n{"k": "release", "rid": "z"}\n')
    with pytest.raises(ValueError):
        led.totals()


def test_complete_tail_missing_newline_kept_on_next_write(tmp_path):
    path = tmp_path / "l.jsonl"
    led = Ledger(path, 1000)
    led.record("s", "x", 10, 0)
    row = {"k": "usage", "rid": None, "scenario": "s", "label": "y",
           "input_tokens": 7, "output_tokens": 0, "cached_tokens": 0}
    with open(path, "a") as f:
        f.write(json.dumps(row))  # complete JSON, no newline
    assert led.totals()["s"]["non_cached"] == 17
    led.record("s", "z", 3, 0)
    assert led.totals()["s"]["non_cached"] == 20
    assert path.read_text().count("\n") == 4  # caps + x + y + z


def test_client_disconnect_mid_stream_settles_hold(env):
    led, p, up = env("sse")
    body = {**REQ, "stream": True}
    payload = json.dumps(body).encode()
    s = socket.create_connection(("127.0.0.1", p.port))
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nX-Live-Label: d\r\nContent-Length: "
              + str(len(payload)).encode() + b"\r\n\r\n" + payload)
    buf = b""
    while b"data:" not in buf:
        chunk = s.recv(4096)
        assert chunk
        buf += chunk
    s.close()
    deadline = time.time() + 10
    while not p.requests("d") and time.time() < deadline:
        time.sleep(0.05)
    assert p.requests("d")
    assert _pending(led) == {}
    assert led.totals()["global"]["non_cached"] in (90, est_of(body))


def test_caps_mismatch_raises(tmp_path):
    Ledger(tmp_path / "l.jsonl", 1000, {"a": 1})
    Ledger(tmp_path / "l.jsonl", 1000, {"a": 1})
    with pytest.raises(ValueError):
        Ledger(tmp_path / "l.jsonl", 999, {"a": 1})


def _worker(path):
    led = Ledger(path, 1000, {})
    for _ in range(20):
        try:
            rid = led.reserve("s", 50)
        except BudgetRefused:
            continue
        led.record("s", "w", 50, 0, 0, rid=rid)


def test_two_processes_respect_cap(tmp_path):
    path = tmp_path / "l.jsonl"
    Ledger(path, 1000, {})
    ps = [multiprocessing.Process(target=_worker, args=(str(path),)) for _ in range(2)]
    [p.start() for p in ps]
    [p.join(60) for p in ps]
    assert all(p.exitcode == 0 for p in ps)
    led = Ledger(path, 1000, {})
    assert led.totals()["global"]["non_cached"] == 1000
    assert led.warn()
