import os, sys, types
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CITEVIZOR_DISABLE_GUIDED", "1")
os.environ.setdefault("SUMMARIZER_JSON_STRICT", "0")
m_pkg = types.ModuleType("pyairports")
m_mod = types.ModuleType("pyairports.airports")
try:
    from airportsdata import load
    _db = load()
    m_mod.AIRPORT_LIST = [
        {"iata": code, "name": meta.get("name",""), "city": meta.get("city",""), "country": meta.get("country","")}
        for code, meta in _db.items()
    ]
except Exception:
    m_mod.AIRPORT_LIST = [
        {"iata": "NRT", "name": "Narita International", "city": "Tokyo", "country": "JP"},
        {"iata": "HKG", "name": "Hong Kong International", "city": "Hong Kong", "country": "HK"},
        {"iata": "PVG", "name": "Shanghai Pudong", "city": "Shanghai", "country": "CN"}
    ]
sys.modules["pyairports"] = m_pkg
sys.modules["pyairports.airports"] = m_mod
