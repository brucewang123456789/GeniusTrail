#!/usr/bin/env python3
# =============================================================================
# demo_toiletpaper_run.py
#
# Single-sample walkthrough (toilet-paper roll) through the SDP pipeline,
# logging: TSR, TE, Latency, Throughput, Chamfer-proxy.
# Pure Python 3.13, zero external dependencies.
# =============================================================================

from __future__ import annotations
import json, math, random, time, uuid, sys
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

# -----------------------------------------------------------------------------
# 1.  AUTO‐DISCOVER & LOAD THE CORRECT PIPELINE MODULE
# -----------------------------------------------------------------------------
HERE = Path(__file__).parent
demo_name = Path(__file__).name

# find all .py files beside this demo
candidates = [p for p in HERE.iterdir() 
              if p.suffix == ".py" and p.name != demo_name]

ped = None
for candidate in candidates:
    spec = spec_from_file_location("ped_sdp", str(candidate))
    if spec and spec.loader:
        mod = module_from_spec(spec)
        sys.modules["ped_sdp"] = mod
        spec.loader.exec_module(mod)  # type: ignore
        # check it has the classes we need
        if hasattr(mod, "SDPPipeline") and hasattr(mod, "PointCloud"):
            ped = mod
            break

if ped is None:
    raise ImportError(
        "Could not find a pipeline module defining SDPPipeline & PointCloud "
        f"in {HERE}. Ensure your architecture .py sits next to this demo."
    )

# now we have the right module
SDPPipeline = ped.SDPPipeline
PointCloud  = ped.PointCloud

# -----------------------------------------------------------------------------
# 2.  MAKE A SYNTHETIC POINT-CLOUD (“toilet-paper roll”)
# -----------------------------------------------------------------------------
def make_tp_roll(n: int = 200) -> PointCloud:
    pts = []
    r_i, r_o, h = 22.0, 27.0, 100.0
    for _ in range(n):
        θ = random.random() * 2 * math.pi
        r = r_i + random.random() * (r_o - r_i)
        z = random.random() * h
        x = r * math.cos(θ)
        y = r * math.sin(θ)
        pts.append((x, y, z))
    return PointCloud(points=pts, meta={"class": "tp_roll"})

# -----------------------------------------------------------------------------
# 3.  QUICK CHAMFER‐PROXY FOR DEMO
# -----------------------------------------------------------------------------
def centroid(pts: list[tuple[float,float,float]]) -> tuple[float,float,float]:
    if not pts: 
        return (0.0,0.0,0.0)
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    sz = sum(p[2] for p in pts)
    n = len(pts)
    return (sx/n, sy/n, sz/n)

def chamfer_proxy(inp: list[tuple[float,float,float]], dec: list[float]) -> float:
    cx,cy,cz = centroid(inp)
    groups = []
    limit = min(len(dec)-2, 30)
    for i in range(0, limit, 3):
        groups.append((dec[i], dec[i+1], dec[i+2]))
    rcx,rcy,rcz = centroid(groups) if groups else (0.0,0.0,0.0)
    return math.sqrt((cx-rcx)**2 + (cy-rcy)**2 + (cz-rcz)**2)

# -----------------------------------------------------------------------------
# 4.  RUN ONE SAMPLE THROUGH A→B→C & COLLECT METRICS
# -----------------------------------------------------------------------------
random.seed(0)
tp_cloud = make_tp_roll()

pipeline = SDPPipeline()
t0 = time.time()
try:
    pipeline.run_once([tp_cloud])
    TSR = 1
except Exception as e:
    print("Pipeline error:", e)
    TSR = 0
t1 = time.time()

TE       = t1 - t0                       # Task Efficiency (s)
Through  = (1.0 / TE) if TE > 0 else 0.0 # Throughput (ops/s)
Latency  = getattr(pipeline, "last_latency", TE)  # from your pipeline or fallback

# get a small reconstruction list (replace with your real API)
if hasattr(pipeline, "reconstruct"):
    recon_vals = pipeline.reconstruct([0] * 32)
else:
    recon_vals = [0.0] * 32

CD = chamfer_proxy(tp_cloud.points, recon_vals)

# -----------------------------------------------------------------------------
# 5.  PRINT JSON REPORT
# -----------------------------------------------------------------------------
report = {
    "sample_id":       str(uuid.uuid4()),
    "TaskSuccessRate": TSR,
    "TaskEfficiency":  round(TE, 4),
    "Latency":         round(Latency, 4),
    "Throughput":      round(Through, 3),
    "ChamferProxy":    round(CD, 4),
}
print(json.dumps(report, indent=2))
