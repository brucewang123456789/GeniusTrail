# /workspace/align/metadata_enricher.py
# -*- coding: utf-8 -*-
"""
Dispatcher shim for MetadataEnricher.

Stable import path for callers:
    from align.metadata_enricher import MetadataEnricher

Actual implementations may live in:
    - retrieval/metadata_enrich.py        (your repo's current layout)
    - retrieval/metadata.py / meta_enricher.py / enricher.py
    - align/metadata_enrich.py            (future refactors, optional)
"""

from importlib import import_module

# Try retrieval.* first (current reality), then align.* as fallback
_CANDIDATE_MODULES = (
    "retrieval.metadata_enrich",
    "retrieval.metadata",
    "retrieval.meta_enricher",
    "retrieval.enricher",
    "align.metadata_enrich",     # optional future location
    "align.metadata",            # optional future location
    "align.meta_enricher",       # optional future location
    "align.enricher",            # optional future location
)
_CANDIDATE_CLASSES = (
    "MetadataEnricher",
    "MetaDataEnricher",
    "Enricher",
)

_last_err = None
MetadataEnricher = None  # type: ignore[assignment]

for mod_name in _CANDIDATE_MODULES:
    try:
        mod = import_module(mod_name)
    except Exception as e:
        _last_err = e
        continue
    for cls_name in _CANDIDATE_CLASSES:
        cls = getattr(mod, cls_name, None)
        if cls is not None:
            MetadataEnricher = cls  # type: ignore[assignment]
            break
    if MetadataEnricher is not None:
        break

if MetadataEnricher is None:  # pragma: no cover
    raise ImportError(
        "align.metadata_enricher: cannot locate implementation; tried modules: "
        f"{_CANDIDATE_MODULES} with classes { _CANDIDATE_CLASSES }.\n"
        "Hint: ensure `retrieval/__init__.py` exists and that "
        "`retrieval/metadata_enrich.py` defines class `MetadataEnricher`."
    ) from _last_err

__all__ = ["MetadataEnricher"]
