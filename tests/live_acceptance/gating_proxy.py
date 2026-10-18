"""OpenAI-compatible gating/metering proxy for live acceptance runs.

Streaming: stream=true is passed through as SSE. stream_options.include_usage
is injected; chunks are relayed as they arrive (flushed); the last chunk with a
usage object is recorded. If upstream answers 400 mentioning stream_options the
request is retried once without it and the estimate is recorded. Gates are
applied before any byte goes to the client.

Spend rules: a hold is reserved before sending. Missing/null/malformed usage or
an ambiguous failure after sending (timeout, reset, error status) records the
ESTIMATE; the hold is released only on a clean connect failure before sending.
Estimate = ceil(len(json(request))/3) + max_tokens (or max_completion_tokens);
without either, default_max_output (default 8192) is assumed, or, with
require_bound=True, the request is refused (400 unbounded).
Worst-case cap overshoot = concurrency x (actual - estimate); see spend_ledger.

The upstream key is read from env and never logged.
"""
import json
import math
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .spend_ledger import BudgetRefused


class _ConnectFail(Exception):
    pass


class GatingProxy:
    def __init__(self, port, log_path, ledger, scenario_getter, upstream_url=None, upstream_key_env=None,
                 mode="forward", canned=None, label_fn=None, default_max_output=8192, require_bound=False,
                 gate_timeout=900, upstream_timeout=300):
        if mode == "forward" and upstream_key_env and not os.environ.get(upstream_key_env):
            raise RuntimeError(f"upstream key env {upstream_key_env} is not set")
        self.log_path, self.ledger, self.scenario_getter = str(log_path), ledger, scenario_getter
        self.upstream_url, self.key_env = upstream_url, upstream_key_env
        self.mode, self.canned, self.label_fn = mode, canned or {}, label_fn
        self.default_max_output, self.require_bound = default_max_output, require_bound
        self.gate_timeout, self.upstream_timeout = gate_timeout, upstream_timeout
        self._gates, self._in_flight, self._unexpected = set(), {}, []
        self._stopped = False
        self.fatal_error = None  # set on upstream 401/403; all later requests are refused
        self._cv = threading.Condition()
        self._log_lock = threading.Lock()
        proxy = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                proxy._handle(self)

        self._server = ThreadingHTTPServer(("127.0.0.1", port), H)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}/v1"

    def start(self):
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def stop(self):
        with self._cv:
            self._stopped = True
            self._gates.clear()
            self._cv.notify_all()
        self._server.shutdown()
        self._server.server_close()

    # -- gates
    def hold(self, name):
        with self._cv:
            self._gates.add(name)

    def release(self, name):
        with self._cv:
            self._gates.discard(name)
            self._cv.notify_all()

    def wait_in_flight(self, name, timeout):
        end = time.monotonic() + timeout
        with self._cv:
            while not self._in_flight.get(name):
                left = end - time.monotonic()
                if left <= 0:
                    return False
                self._cv.wait(left)
            return True

    # -- logs
    def requests(self, label=None):
        with self._log_lock, open(self.log_path, "a+") as f:
            f.seek(0)
            rows = [json.loads(x) for x in f.read().splitlines() if x.strip()]
        return [r for r in rows if label is None or r["label"] == label]

    def unexpected_calls(self):
        return list(self._unexpected)

    def _log(self, **row):
        row["ts"] = time.time()
        with self._log_lock, open(self.log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    # -- helpers
    @staticmethod
    def _reply(h, status, body):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        h.send_response(status)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", str(len(data)))
        h.end_headers()
        h.wfile.write(data)

    @classmethod
    def _reply_safe(cls, h, status, body):
        try:
            cls._reply(h, status, body)
        except Exception:
            pass

    def _estimate(self, req):
        out = req.get("max_tokens") or req.get("max_completion_tokens")
        if out is None:
            if self.require_bound:
                return None
            out = self.default_max_output
        return math.ceil(len(json.dumps(req)) / 3) + int(out)

    def _open(self, body):
        headers = {"Content-Type": "application/json"}
        if self.key_env and os.environ.get(self.key_env):
            headers["Authorization"] = "Bearer " + os.environ[self.key_env]
        up = urllib.request.Request(self.upstream_url.rstrip("/") + "/chat/completions", body, headers)
        try:
            return urllib.request.urlopen(up, timeout=self.upstream_timeout)
        except urllib.error.HTTPError as e:
            return e
        except urllib.error.URLError as e:
            if isinstance(e.reason, (ConnectionRefusedError, socket.gaierror)):  # nothing sent
                raise _ConnectFail() from e
            raise

    def _settle(self, scenario, label, rid, est, usage):
        """Record actual usage if well-formed, else the estimate. Returns the logged usage."""
        if isinstance(usage, dict) and usage.get("prompt_tokens") is not None \
                and usage.get("completion_tokens") is not None:
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            try:
                self.ledger.record(scenario, label, usage["prompt_tokens"], usage["completion_tokens"],
                                   cached, rid=rid)
                return {"input": usage["prompt_tokens"], "output": usage["completion_tokens"],
                        "cached": int(cached or 0), "estimated": False}
            except (ValueError, TypeError):
                pass
        self.ledger.record(scenario, label, est, 0, 0, rid=rid)
        return {"input": est, "output": 0, "cached": 0, "estimated": True}

    @staticmethod
    def _body_usage(data):
        try:
            return json.loads(data).get("usage")
        except (ValueError, AttributeError):
            return None

    # -- request handling
    def _handle(self, h):
        path = h.path
        raw = h.rfile.read(int(h.headers.get("Content-Length", "0")))
        if self._stopped:
            return self._reply(h, 503, {"error": {"reason": "stopped"}})
        if self.fatal_error:
            return self._reply(h, 503, {"error": {"reason": "fatal_upstream_auth"}})
        if not path.rstrip("/").endswith("/chat/completions"):
            return self._reply(h, 404, {"error": {"reason": "not_found"}})
        try:
            req = json.loads(raw)
        except ValueError:
            return self._reply(h, 400, {"error": {"reason": "bad_json"}})
        label = (self.label_fn(req, h.headers) if self.label_fn else None) or h.headers.get("X-Live-Label") or "unlabeled"

        gate = None
        with self._cv:
            gate = label if label in self._gates else ("*" if "*" in self._gates else None)
            if gate:
                keys = {gate, label}
                for k in keys:
                    self._in_flight[k] = self._in_flight.get(k, 0) + 1
                self._cv.notify_all()
                deadline = time.monotonic() + self.gate_timeout
                while gate in self._gates and not self._stopped:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        break
                    self._cv.wait(min(left, 0.1))
                timed_out = gate in self._gates and not self._stopped
                for k in keys:
                    self._in_flight[k] -= 1
        if gate and timed_out:
            self._log(path=path, label=label, gate=gate, status=504, usage=None)
            return self._reply(h, 504, {"error": {"reason": "gate_timeout"}})
        if self._stopped:
            return self._reply(h, 503, {"error": {"reason": "stopped"}})

        if self.mode == "canned":
            if label in self.canned:
                self._log(path=path, label=label, gate=gate, status=200, usage=None)
                return self._reply(h, 200, self.canned[label])
            self._unexpected.append({"label": label, "path": path})
            self._log(path=path, label=label, gate=gate, status=500, usage=None, unexpected_call=True)
            return self._reply(h, 500, {"error": {"reason": "unexpected_call"}})

        est = self._estimate(req)
        if est is None:
            return self._reply(h, 400, {"error": {"reason": "unbounded"}})
        scenario = self.scenario_getter()
        try:
            rid = self.ledger.reserve(scenario, est)
        except BudgetRefused:
            self._log(path=path, label=label, gate=gate, status=429, usage=None)
            return self._reply(h, 429, {"error": {"reason": "budget"}})
        self._forward(h, req, raw, path, label, gate, scenario, rid, est)

    def _forward(self, h, req, raw, path, label, gate, scenario, rid, est):
        stream = bool(req.get("stream"))
        body, injected = raw, False
        if stream:
            so = dict(req.get("stream_options") or {})
            if not so.get("include_usage"):
                so["include_usage"] = True
                body, injected = json.dumps({**req, "stream_options": so}).encode(), True
        usage, status, sent = None, 502, False
        fatal, data, opened = False, b"", []
        try:
            try:
                r = self._open(body)
                opened.append(r)
                status = getattr(r, "status", None) or r.code
                if status == 400 and injected:
                    data = r.read()
                    if b"stream_options" in data:
                        r = self._open(raw)  # retry once without; usage then unknown -> estimate
                        opened.append(r)
                        status = getattr(r, "status", None) or r.code
                    else:
                        r = _Replay(r, data)
            except _ConnectFail:
                self.ledger.release(rid)  # nothing was sent
                self._log(path=path, label=label, gate=gate, status=502, usage=None)
                return self._reply_safe(h, 502, {"error": {"reason": "upstream_unreachable"}})
            if status in (401, 403):
                fatal, data = True, r.read()
            elif stream and status < 400:
                h.send_response(status)
                h.send_header("Content-Type", r.headers.get("Content-Type", "text/event-stream"))
                h.send_header("Cache-Control", "no-cache")
                h.end_headers()  # no Content-Length: body ends when connection closes
                sent = True
                for line in iter(r.readline, b""):
                    h.wfile.write(line)
                    h.wfile.flush()
                    if line.startswith(b"data:") and b"[DONE]" not in line:
                        try:
                            u = json.loads(line[5:]).get("usage")
                        except (ValueError, AttributeError):
                            u = None
                        if isinstance(u, dict):
                            usage = u
                h.close_connection = True
            else:
                data = r.read()
                usage = self._body_usage(data)
                self._reply(h, status, data)
                sent = True
        except Exception:
            pass  # ambiguous after send: fall through, estimate is recorded
        finally:
            for x in opened:
                try:
                    x.close()
                except Exception:
                    pass
        if fatal:
            self.ledger.record(scenario, label, 0, 0, 0, rid=rid)  # nothing billed; settles the hold
            self.fatal_error = {"status": status, "ts": time.time()}
            self._log(path=path, label=label, gate=gate, status=status, usage=None)
            return self._reply_safe(h, status, data)
        logged = self._settle(scenario, label, rid, est, usage)
        self._log(path=path, label=label, gate=gate, status=status, usage=logged)
        if not sent:
            self._reply_safe(h, 502, {"error": {"reason": "upstream_failed"}})


class _Replay:
    """Re-serves an already-read HTTPError body."""
    def __init__(self, r, data):
        self.code, self.status, self.headers, self._d = r.code, r.code, r.headers, data

    def read(self):
        return self._d
