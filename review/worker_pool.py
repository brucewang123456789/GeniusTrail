# C:\CiteVizor\pipeline\worker_pool.py
# -*- coding: utf-8 -*-
"""
CiteVizor - Generic concurrent worker pool for IO-bound tasks

Design goals
- Bounded concurrency via ThreadPoolExecutor (IO-friendly)
- Per-task retry with exponential backoff + jitter
- Optional per-task timeout (wait timeout; thread cannot be force-killed)
- Optional global rate limiting (QPS) to protect remote services
- Order-preserving results (if requested) or as-completed streaming
- Unified structured logging (run_id propagation, spans, error details)

This module has NO third-party deps and is safe on Windows/Linux/macOS.
It integrates with infra.logging if present; otherwise it degrades to no-op logs.

Usage (quick example)
---------------------
from pipeline.worker_pool import WorkerPool, TaskSpec

def fetch(url: str) -> str:
    return requests.get(url, timeout=8).text[:200]

pool = WorkerPool(max_workers=8, rate_qps=4.0, run_id="run_123")
tasks = [TaskSpec(id=f"u{i}", func=fetch, args=(u,), timeout_s=10.0, retries=2) for i,u in enumerate(urls)]
results = pool.run(tasks, preserve_order=True)
for r in results:
    if r.ok: print(r.id, len(r.value))
    else:    print(r.id, "ERR", r.error)

"""

from __future__ import annotations

import time
import random
import threading
import contextvars
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# ----------------------------- logging (safe fallback) -------------------------
try:
    from infra.logging import setup_logging, get_logger, set_run_id, span, log_event  # type: ignore
except Exception:  # graceful fallback: no-op logging
    def setup_logging(*args, **kwargs):  # type: ignore
        return None
    def get_logger(name: Optional[str] = None):  # type: ignore
        class _L:
            def debug(self, *a, **k): pass
            def info(self, *a, **k): pass
            def warning(self, *a, **k): pass
            def error(self, *a, **k): pass
        return _L()
    def set_run_id(*args, **kwargs):  # type: ignore
        return None
    class span:  # type: ignore
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): return False
    def log_event(*args, **kwargs):  # type: ignore
        return None

LOG = get_logger("worker_pool")

# ----------------------------- rate limiter ------------------------------------

class RateLimiter:
    """
    Simple token bucket limiter: allow ~qps events, with burst <= bucket.
    """
    def __init__(self, qps: Optional[float], bucket: Optional[int] = None):
        self.qps = qps if (qps and qps > 0) else None
        if self.qps is None:
            self.bucket = 0
            self.tokens = 0.0
        else:
            self.bucket = bucket or max(1, int(self.qps * 2))
            self.tokens = float(self.bucket)
        self._lock = threading.Lock()
        self._last = time.perf_counter()

    def acquire(self) -> None:
        if self.qps is None:
            return
        with self._lock:
            now = time.perf_counter()
            elapsed = max(0.0, now - self._last)
            self._last = now
            self.tokens = min(self.bucket, self.tokens + elapsed * self.qps)
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return
            # need to wait until enough tokens accumulate
            need = 1.0 - self.tokens
            delay = need / self.qps
        # sleep outside lock
        time.sleep(delay)
        # second attempt (recursive but shallow)
        self.acquire()

# ----------------------------- datamodel ---------------------------------------

@dataclass
class TaskSpec:
    """
    Describe a single unit of work.
    """
    id: str
    func: Callable[..., Any]
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    timeout_s: Optional[float] = None
    retries: int = 0
    backoff_base_s: float = 0.5       # initial backoff
    backoff_factor: float = 2.0       # exponential growth
    backoff_max_s: float = 8.0        # cap
    kind: str = "task"                # e.g., "download", "parse", "http"
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskResult:
    id: str
    ok: bool
    value: Any = None
    error: Optional[str] = None
    exc_type: Optional[str] = None
    attempts: int = 0
    started_at: float = 0.0
    ended_at: float = 0.0
    duration_s: float = 0.0
    kind: str = "task"
    meta: Dict[str, Any] = field(default_factory=dict)

# ----------------------------- worker pool -------------------------------------

class WorkerPool:
    """
    A thin wrapper around ThreadPoolExecutor with retries, rate limiting, and structured logging.
    Thread-based because our pipeline is IO-bound (HTTP, file I/O, PDF parsing).
    """

    def __init__(
        self,
        max_workers: int = 8,
        rate_qps: Optional[float] = None,
        run_id: Optional[str] = None,
        name: str = "pool",
    ):
        self.max_workers = max(1, int(max_workers))
        self.exec = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix=f"{name}-t")
        self.limiter = RateLimiter(rate_qps)
        self.name = name
        self._closed = False
        if run_id:
            try:
                set_run_id(run_id)
            except Exception:
                pass
        LOG.info("worker_pool.init", extra={"name": self.name, "max_workers": self.max_workers, "rate_qps": rate_qps})

    # ---- internal helpers ----

    @staticmethod
    def _sleep_backoff(attempt: int, base: float, factor: float, max_s: float) -> float:
        delay = min(max_s, base * (factor ** max(0, attempt - 1)))
        # small jitter (0~250ms) to avoid thundering herd
        jitter = random.random() * 0.25
        return delay + jitter

    def _call_once(self, spec: TaskSpec) -> Any:
        """
        Call the user's function once (without retries), respecting rate limit.
        """
        # rate limit prior to call
        self.limiter.acquire()
        return spec.func(*spec.args, **spec.kwargs)

    def _run_with_retries(self, spec: TaskSpec) -> TaskResult:
        """
        Execute a TaskSpec with retries and logging; return TaskResult.
        """
        result = TaskResult(id=spec.id, ok=False, kind=spec.kind, meta=spec.meta)
        result.started_at = time.time()

        last_err: Optional[Exception] = None
        attempts = 0
        # Make a local logger with task kind
        logger = get_logger(f"{self.name}.{spec.kind}")

        while True:
            attempts += 1
            with span(logger, f"{spec.kind}", task_id=spec.id, attempt=attempts):
                try:
                    # call once
                    value = self._call_once(spec)
                    # optionally wait with timeout for future? (we're already inside thread)
                    result.value = value
                    result.ok = True
                    break
                except Exception as e:
                    last_err = e
                    logger.warning("task.attempt_failed", extra={
                        "task_id": spec.id, "attempt": attempts, "exc": type(e).__name__, "msg": str(e)[:500]
                    })
                    # retry policy
                    if attempts > max(1, int(spec.retries)) + 1 - 1:  # attempts includes first try
                        break
                    delay = self._sleep_backoff(attempts, spec.backoff_base_s, spec.backoff_factor, spec.backoff_max_s)
                    time.sleep(delay)

        result.attempts = attempts
        result.ended_at = time.time()
        result.duration_s = round(result.ended_at - result.started_at, 3)
        if not result.ok:
            result.error = str(last_err)[:1000] if last_err else "unknown error"
            result.exc_type = type(last_err).__name__ if last_err else None
        logger.info("task.done", extra={
            "task_id": spec.id, "ok": result.ok, "attempts": attempts, "elapsed_s": result.duration_s,
            "exc": result.exc_type if not result.ok else ""
        })
        return result

    # ---- public API ----

    def submit(self, spec: TaskSpec) -> Future:
        """
        Submit a single task. Returns a Future of TaskResult.
        NOTE: timeout_s is best-effort; threads cannot be force-killed.
        """
        if self._closed:
            raise RuntimeError("WorkerPool already closed")

        # Run the retry loop inside the worker thread
        fut: Future = self.exec.submit(self._run_with_retries, spec)

        # Wrap waiting with timeout by attaching a small proxy future if required
        if spec.timeout_s and spec.timeout_s > 0:
            proxy: Future = self.exec.submit(self._await_with_timeout, fut, spec.timeout_s, spec)
            return proxy
        return fut

    @staticmethod
    def _await_with_timeout(inner: Future, timeout_s: float, spec: TaskSpec) -> TaskResult:
        """
        Wait for 'inner' future with timeout. If timeout occurs, mark as failure
        and leave the original thread to finish in background (best effort).
        """
        try:
            return inner.result(timeout=timeout_s)
        except Exception as e:
            # TimeoutError or other errors while waiting
            tr = TaskResult(id=spec.id, ok=False, kind=spec.kind, meta=spec.meta)
            tr.started_at = time.time()
            tr.ended_at = tr.started_at
            tr.duration_s = 0.0
            tr.error = f"timeout/{type(e).__name__}: {e}"
            tr.exc_type = type(e).__name__
            tr.attempts = 1
            # Try to cancel if not running
            inner.cancel()
            LOG.error("task.timeout", extra={"task_id": spec.id, "timeout_s": timeout_s})
            return tr

    def run(self, specs: Sequence[TaskSpec], preserve_order: bool = False,
            on_progress: Optional[Callable[[TaskResult], None]] = None) -> List[TaskResult]:
        """
        Run a batch of tasks and return results. If preserve_order=True, results follow 'specs' order.
        """
        if not specs:
            return []

        # initial submit
        futures: List[Future] = []
        id_to_index: Dict[str, int] = {}
        for idx, s in enumerate(specs):
            id_to_index[s.id] = idx
            futures.append(self.submit(s))
        LOG.info("batch.submitted", extra={"name": self.name, "count": len(futures)})

        results: List[Optional[TaskResult]] = [None] * len(specs) if preserve_order else []
        done_count = 0

        for fut in as_completed(futures):
            try:
                r: TaskResult = fut.result()
            except Exception as e:
                # Should not happen because _run_with_retries already captures exceptions,
                # but keep a guard here.
                r = TaskResult(id="unknown", ok=False, error=str(e), exc_type=type(e).__name__)

            if preserve_order:
                idx = id_to_index.get(r.id, None)
                if idx is not None:
                    results[idx] = r
                else:
                    results.append(r)  # unexpected id; append
            else:
                results.append(r)

            done_count += 1
            if on_progress:
                try:
                    on_progress(r)
                except Exception:
                    pass

        # fill missing if any
        if preserve_order:
            results = [r if r is not None else TaskResult(id=specs[i].id, ok=False, error="missing result")
                       for i, r in enumerate(results)]
        LOG.info("batch.done", extra={
            "name": self.name,
            "count": len(results),
            "ok": sum(1 for r in results if r.ok),
            "fail": sum(1 for r in results if not r.ok),
        })
        return results

    def shutdown(self, wait: bool = True, cancel_futures: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.exec.shutdown(wait=wait, cancel_futures=cancel_futures)
        finally:
            LOG.info("worker_pool.closed", extra={"name": self.name, "wait": wait, "cancel": cancel_futures})


# ----------------------------- convenience APIs --------------------------------

def run_map(
    func: Callable[..., Any],
    iterable: Iterable[Any],
    *,
    arg_mode: str = "single",        # "single" -> func(x); "tuple" -> func(*x); "kw" -> func(**x)
    max_workers: int = 8,
    rate_qps: Optional[float] = None,
    timeout_s: Optional[float] = None,
    retries: int = 0,
    kind: str = "task",
    mk_task_id: Optional[Callable[[Any, int], str]] = None,
    run_id: Optional[str] = None,
    preserve_order: bool = True,
) -> List[TaskResult]:
    """
    Quick map helper that wraps iterables into TaskSpec and runs them via WorkerPool.
    """
    pool = WorkerPool(max_workers=max_workers, rate_qps=rate_qps, run_id=run_id, name=kind)
    specs: List[TaskSpec] = []
    for i, x in enumerate(iterable):
        tid = mk_task_id(x, i) if mk_task_id else f"{kind}_{i:06d}"
        if arg_mode == "tuple":
            args, kwargs = tuple(x), {}
        elif arg_mode == "kw":
            args, kwargs = (), dict(x)
        else:
            args, kwargs = (x,), {}
        specs.append(TaskSpec(
            id=tid, func=func, args=args, kwargs=kwargs,
            timeout_s=timeout_s, retries=retries, kind=kind
        ))
    try:
        return pool.run(specs, preserve_order=preserve_order)
    finally:
        pool.shutdown()

# ----------------------------- self-test ---------------------------------------

if __name__ == "__main__":
    # Minimal self-test: 10 fake tasks with random failure and rate limit
    import os
    setup_logging(level=os.getenv("LOG_LEVEL", "INFO"))
    def fake_job(x: int) -> str:
        # simulate variable latency and sporadic errors
        t = 0.05 + random.random() * 0.15
        time.sleep(t)
        if random.random() < 0.15:
            raise RuntimeError("random failure")
        return f"ok:{x}"

    items = list(range(10))
    results = run_map(fake_job, items, max_workers=4, rate_qps=8.0, retries=2, kind="demo")
    print("---- SUMMARY ----")
    for r in results:
        print(r.id, "OK" if r.ok else "ERR", r.value if r.ok else r.error)
