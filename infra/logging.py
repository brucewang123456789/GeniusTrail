# infra/logging.py
from __future__ import annotations
import gzip, json, os, sys, time, uuid, contextvars
from pathlib import Path
from typing import Any, Dict, Optional
import logging as py_logging
from logging.handlers import RotatingFileHandler
try:
    from schemas import STORAGE_DIR  # type: ignore
except Exception:
    from pathlib import Path as _P
    STORAGE_DIR = _P.cwd() / "storage"
try:
    from config import load_config  # type: ignore
except Exception:
    load_config = None  # type: ignore

RUN_ID_CVAR: contextvars.ContextVar[str] = contextvars.ContextVar("run_id", default="")
REQ_ID_CVAR: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")

def set_run_id(run_id: str) -> None:
    RUN_ID_CVAR.set(str(run_id or ""))

def get_run_id() -> str:
    return RUN_ID_CVAR.get()

def set_request_id(req_id: Optional[str] = None) -> str:
    rid = req_id or uuid.uuid4().hex[:16]
    REQ_ID_CVAR.set(rid)
    return rid

def get_request_id() -> str:
    return REQ_ID_CVAR.get()

def _env_lookup(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(key)
    if v is not None:
        return v
    if load_config:
        try:
            cfg = load_config()
            raw = getattr(cfg, "raw", {}) or {}
            if key in raw:
                return raw[key]
        except Exception:
            pass
    return default

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

DEFAULT_KEYS = {
    "name","msg","args","levelname","levelno","pathname","filename","module",
    "exc_info","exc_text","stack_info","lineno","funcName","created","msecs",
    "relativeCreated","thread","threadName","processName","process",
}

class JSONFormatter(py_logging.Formatter):
    def format(self, record: py_logging.LogRecord) -> str:
        base: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "lvl": record.levelname,
            "name": record.name,
            "msg": str(record.getMessage()).replace("\n"," ").replace("\r"," ").strip(),
            "run_id": getattr(record, "run_id", "") or get_run_id(),
            "request_id": getattr(record, "request_id", "") or get_request_id(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }
        for k, v in record.__dict__.items():
            if k not in DEFAULT_KEYS and k not in base and not k.startswith("_"):
                try:
                    json.dumps(v)
                    base[k] = v
                except Exception:
                    base[k] = repr(v)
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info).replace("\n"," | ")
        return json.dumps(base, ensure_ascii=False, separators=(",",":"))

class ConsoleFormatter(py_logging.Formatter):
    COLORS = {"DEBUG":"\033[37m","INFO":"\033[36m","WARNING":"\033[33m","ERROR":"\033[31m","CRITICAL":"\033[41m"}
    RESET = "\033[0m"
    def __init__(self, use_color: bool = True):
        super().__init__()
        self.use_color = use_color and sys.stderr.isatty()
    def format(self, record: py_logging.LogRecord) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        lvl = record.levelname
        color = self.COLORS.get(lvl,"") if self.use_color else ""
        reset = self.RESET if self.use_color else ""
        rid = getattr(record, "run_id", "") or get_run_id()
        qid = getattr(record, "request_id", "") or get_request_id()
        msg = str(record.getMessage()).replace("\n"," ").replace("\r"," ").strip()
        extras = []
        for k,v in record.__dict__.items():
            if k not in DEFAULT_KEYS and k not in {"run_id","request_id"} and not k.startswith("_"):
                try:
                    s = json.dumps(v, ensure_ascii=False)
                except Exception:
                    s = repr(v)
                extras.append(f"{k}={s}")
        if rid: extras.append(f"run={rid}")
        if qid: extras.append(f"req={qid}")
        extra_str = (" " + " ".join(extras)) if extras else ""
        return f"{color}[{ts} {lvl:<7}]{reset} {record.name}:{record.lineno} - {msg}{extra_str}"

class GZipRotatingFileHandler(RotatingFileHandler):
    def doRollover(self) -> None:
        super().doRollover()
        try:
            bkp = f"{self.baseFilename}.1"
            if os.path.exists(bkp):
                import gzip as _gz
                with open(bkp,"rb") as f_in, _gz.open(f"{bkp}.gz","wb",compresslevel=6) as f_out:
                    f_out.writelines(f_in)
                os.remove(bkp)
        except Exception:
            pass

def setup_logging(
    level: Optional[str] = None,
    json_console: Optional[bool] = None,
    log_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
    file_json: bool = True,
    file_name: str = "citevizor.log",
    file_max_mb: int = 10,
    file_backups: int = 7,
) -> None:
    level = (level or _env_lookup("LOG_LEVEL","INFO")).upper()
    json_console = bool(int(_env_lookup("LOG_JSON","0"))) if json_console is None else bool(json_console)
    log_dir = log_dir or Path(_env_lookup("LOG_DIR", str(STORAGE_DIR / "logs")))
    if run_id:
        set_run_id(run_id)
    _ensure_dir(log_dir)
    root = py_logging.getLogger()
    if getattr(root, "_citevizor_configured", False):
        root.setLevel(level)
        if run_id:
            set_run_id(run_id)
        return
    root.setLevel(level)
    ch = py_logging.StreamHandler(stream=sys.stderr)
    ch.setLevel(level)
    ch.setFormatter(JSONFormatter() if json_console else ConsoleFormatter())
    root.addHandler(ch)
    fh_path = log_dir / file_name
    fh = GZipRotatingFileHandler(
        filename=str(fh_path),
        maxBytes=file_max_mb*1024*1024,
        backupCount=file_backups,
        encoding="utf-8",
    )
    fh.setLevel(level)
    fh.setFormatter(JSONFormatter() if file_json else ConsoleFormatter(use_color=False))
    root.addHandler(fh)
    setattr(root, "_citevizor_configured", True)
    root.info("Logging initialized", extra={
        "run_id": get_run_id(),
        "level": level,
        "log_dir": str(log_dir),
        "json_console": json_console,
        "file": str(fh_path),
    })

def get_logger(name: Optional[str] = None) -> py_logging.Logger:
    return py_logging.getLogger(name or "citevizor")

def log_event(logger: py_logging.Logger, event: str, **fields: Any) -> None:
    fields = dict(fields or {})
    fields.setdefault("run_id", get_run_id())
    fields.setdefault("request_id", get_request_id())
    logger.info(event, extra=fields)

class span:
    def __init__(self, logger: py_logging.Logger, name: str, **fields: Any):
        self.logger = logger
        self.name = name
        self.fields = fields
        self._t0 = 0.0
        self._sid = uuid.uuid4().hex[:12]
    def __enter__(self):
        import time as _t
        self._t0 = _t.perf_counter()
        self.logger.debug(f"[span.start] {self.name}", extra={**self.fields,"span_id":self._sid,"run_id":get_run_id(),"request_id":get_request_id()})
        return self
    def __exit__(self, exc_type, exc, tb):
        import time as _t
        dt = _t.perf_counter() - self._t0
        payload = {**self.fields,"span_id":self._sid,"elapsed_s":round(dt,3),"run_id":get_run_id(),"request_id":get_request_id()}
        if exc_type:
            self.logger.error(f"[span.error] {self.name}: {exc_type.__name__}: {exc}", extra=payload)
            return False
        self.logger.info(f"[span.end] {self.name}", extra=payload)
        return False

def add_fastapi_request_logging(app) -> None:
    try:
        from starlette.middleware.base import BaseHTTPMiddleware
    except Exception:
        return
    logger = get_logger("http")
    class _ReqLogMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            rid = request.headers.get("X-Request-ID") or set_request_id()
            try:
                response = await call_next(request)
                response.headers["X-Request-ID"] = rid
                logger.info("request", extra={
                    "method": request.method,
                    "path": request.url.path,
                    "client": request.client.host if request.client else "",
                    "status": getattr(response, "status_code", 0),
                    "run_id": get_run_id(),
                    "request_id": rid,
                })
                return response
            finally:
                set_request_id("")
    app.add_middleware(_ReqLogMiddleware)

if __name__ == "__main__":
    setup_logging(level="INFO", json_console=False, run_id="demo")
    log = get_logger("demo")
    log.info("one line ok", extra={"phase":"init"})
    with span(log,"sleep", secs=0.01):
        import time; time.sleep(0.01)
