# CiteVizor Backend (Prototype Academic Agent)

CiteVizor is an engineer-facing academic agent backend. It turns a natural-language research query into an **image-first technical report** and an optional **PPTX slide deck**.

This repository currently contains **backend only**: the end-to-end pipeline from Scholar / web retrieval, through PDF download and parsing, figure and table extraction, evidence ranking, LLM summarisation, chart generation, HTML layout, PPTX export, and light-weight QA with run tracing.

The system already runs end-to-end on real papers (for example “Dynamic Chain-of-Thought: Towards Adaptive Deep Reasoning”, several Chain-of-Thought survey queries, and “Attention Is All You Need”), but it is still a **research prototype**: there is no production-grade front-end, no multi-tenant service layer, and no systematic benchmark suite.

# 1. Scope and Current Level of Completion

Implemented backend capabilities

- Single Python entrypoint that runs the complete pipeline from query to HTML + PPTX.
- Retrieval from Google Scholar and the open web via Serper, with aggregation and ranking.
- Predownload URL resolution that tries to turn landing pages, mirrors and DOIs into concrete PDF or image links.
- A download layer with a fixed storage layout that writes PDFs and web images under a stable `storage/` tree.
- PDF parsing into paragraphs, tables and figures, with an optional OCR fallback for image-only pages.
- Figure extraction via `pdffigures2`, integrated into the same evidence pack as text and tables.
- Evidence ranking and citation map construction that keeps explicit links from every highlight back to specific pages, tables or figures.
- Engineer-oriented summariser that runs on a local OpenAI-style vLLM server.
- LLM-driven Matplotlib chart code generation, plus deterministic rendering to PNG.
- HTML report where visuals dominate the left column and key bullet points occupy the right column.
- PPTX exporter that mirrors the HTML structure and places one main visual per slide.
- QA checks and a per-run JSON manifest that record retrieval counts, parsing statistics, chart counts and basic consistency checks.

Explicitly not done yet

- No user-facing web application or front-end project.
- No user accounts, background job queue, permission system or concurrency management.
- Testing is mostly manual end-to-end; there is no large-scale evaluation pipeline yet.
- English is the main target language; multilingual behaviour has not been tuned.
- Error handling aims for “best effort” and graceful degradation, not strict robustness or SLAs.

The honest status: CiteVizor is a **backend research skeleton**. It is usable for demos, experiments and internal tools, but it should not be treated as a finished SaaS product.

# 2. Architecture Overview

Top-level files

- `citevizor_backend.py`  
  Main CLI entrypoint, orchestrates all stages of the pipeline.
- `config.py`  
  Loads environment variables and configuration into a typed Config object.
- `schemas.py`  
  Dataclasses for documents, evidence packs, citations, run metadata, and centralised path helpers.

Core directories

- `planner/` – optional query planning and year-range handling.
- `retrieval/` – Scholar + web search, candidate aggregation, ranking and predownload URL resolution.
- `fetch/` – HTTP downloader and storage policy.
- `parse/` – PDF parsing, table detection, figure integration, OCR fallback.
- `align/` – evidence ranking and citation-map construction.
- `llm/` – vLLM client, engineer summariser and chart code generator.
- `viz/` – Matplotlib rendering and chart deduplication / selection.
- `report/` – HTML layout engine and PPTX exporter.
- `qa/` – basic consistency checks and per-run manifest.
- `review/` – lightweight preview server for browsing artefacts under `storage/`.
- `tools/pdffigures2/` – wrapper and configuration for the Scala `pdffigures2` extractor.
- `storage/` – all runtime artefacts produced by pipeline runs.

# 3. Models and External Services

3.1 LLM runtime

- Engine: local vLLM HTTP server with an OpenAI-compatible API.
- Default model: `Qwen/Qwen2.5-14B-Instruct-AWQ` (quantised).
- API surface: the backend uses only `POST /v1/completions`; it deliberately does not rely on chat endpoints or tool-calling JSON schemas.
- Roles in CiteVizor:
  - engineer-facing scientific summarisation based on explicit evidence;
  - Matplotlib chart code generation from table descriptions.

Model name, decoding parameters and API base URL are provided via environment variables. Any other model that behaves like an OpenAI completion model can be swapped in by configuration only.

3.2 Retrieval layer

- Provider: Serper.
- Scholar channel: queries Google Scholar to obtain academic entries with title, URL, host, year and snippet.
- Web channel: queries the general web to obtain arXiv pages, project pages, blog posts and other high-value resources.
- Aggregation:
  - results are merged into a unified Candidate structure;
  - a reciprocal-rank-style score is used with mild recency bias and host-diversity penalties.

3.3 Parsing and visual tools

- Figure extraction: Scala `pdffigures2` JAR, invoked via a small Python wrapper.
- Text parsing: PyMuPDF and pdfplumber for page layout and text extraction.
- OCR: optional, for image-only PDFs.
- Chart rendering: Matplotlib only, with style libraries intentionally disabled to keep chart code simple and predictable.

# 4. Storage Layout

All artefacts live under `storage/`. Document identifiers `<doc_id>` and run identifiers `<run_id>` appear both in logs and in file paths.

storage/  
  docs/<doc_id>/<doc_id>.pdf  
  docs/<doc_id>/<doc_id>.html  

  evidence/<doc_id>/pack.json  
  evidence/<doc_id>/figures/*.png  
  evidence/<doc_id>/web_images/*.png  
  evidence/_rank/<run_id>.citations.json  
  evidence/_rank/<run_id>.summary.json  

  renders/<doc_id>/charts/chart_*.png  
  renders/<doc_id>/charts/scripts/*.py  
  renders/<doc_id>/charts/_charts.json  

  reports/<run_id>.html  
  reports/<run_id>.pptx  
  reports/_runs/<run_id>.manifest.json  

This fixed layout makes it easy to inspect or replay each stage for a given run using only the filesystem.

5. Pipeline Stages

The full pipeline is orchestrated in `citevizor_backend.py`. Log messages with tags such as “retrieval.plan”, “download.done”, “parse.doc”, “citations.saved”, “export.pptx” and “trace.manifest” correspond directly to the stages described below.

5.1 Query planning (optional)

- Module: `planner/plan_query.py`
- Input: natural-language query plus optional year constraint (for example `>=2025`).
- Behaviour:
  - parse textual year expressions into a concrete year range;
  - derive several Scholar-friendly sub-queries;
  - optionally ask the LLM for additional keyword variants.
- Output: `SearchPlan` JSON in `storage/plans/`.

If query planning is disabled, a minimal SearchPlan is constructed directly from the raw query so that the downstream pipeline still runs.

5.2 Retrieval (Scholar + web)

- Modules: `retrieval/serper_scholar.py`, `retrieval/serper_web.py`, `retrieval/aggregate_rank.py`.
- Behaviour:
  - send Serper Scholar and web requests according to the search plan;
  - parse responses into a common Candidate schema (title, URL, host, year, scores);
  - aggregate and re-rank candidates using relevance, recency and host diversity.
- Output: aggregated list in `storage/retrieval/<run_id>.agg.json`.

These candidates are the starting point for predownload resolution and download.

5.3 Predownload URL resolution

- Module: `retrieval/predownload.py`.
- Behaviour:
  - inspect each candidate URL and attempt to resolve a concrete PDF or image URL from landing pages, arXiv pages or DOI endpoints;
  - assign a coarse type label (`pdf`, `image`, `landing`, `unknown`);
  - rewrite the URL when a better target is discovered.
- Output: enriched candidate list with type hints, passed to the downloader.

This stage is fast and best-effort; failures fall back to the original URL, and statistics are logged.

5.4 Download

- Module: `fetch/downloader.py`.
- Behaviour:
  - download PDFs and images under reasonable timeouts and per-host throttling;
  - verify content type and magic bytes to reduce the chance of saving HTML as PDF;
  - enforce the storage layout for `storage/docs/` and `storage/evidence/<doc_id>/web_images/`;
  - record explicit success / failure reasons for QA and human inspection.
- Output: actual files on disk, plus an in-memory list of `FetchedDoc` instances.

This is where log lines like “download.doc” and “download.done” come from.

5.5 Figure extraction

- Wrapper: `parse/figure_extract.py`.
- Behaviour:
  - call the pdffigures2 JAR for each PDF;
  - convert its JSON output into internal figure records with stable IDs;
  - write PNG files to `storage/evidence/<doc_id>/figures/` and register captions and locations.
- Output: figure evidence referenced later by packs and reports.

This module has been tested both in isolation and as part of the full pipeline.

5.6 PDF parsing, tables and OCR

- Dispatcher: `parse/parse_dispatch.py`.
- Behaviour:
  - use PyMuPDF / pdfplumber to extract page-level text spans and structural hints;
  - detect tables and turn them into explicit row / column structures;
  - merge figure metadata from the previous stage;
  - optionally trigger OCR for image-only pages when the `--enable-ocr` flag is set.
- Output: one `EvidencePack` per document in `storage/evidence/<doc_id>/pack.json`.

The log counts such as “paragraphs=88 tables=0 figures=19” are derived from these packs.

5.7 Evidence ranking and citation map

- Modules: `align/evidence_rank.py`, `align/citation_map.py`.
- Behaviour:
  - build a sparse index over paragraphs, figure captions and table descriptions;
  - score evidence spans against the original query with a BM25-style scheme;
  - select a compact set of spans that best characterise the paper for the query;
  - assign human-readable reference labels such as “[1, p.7]” or “[2, Fig.3]”.
- Output: citation map stored as `storage/evidence/_rank/<run_id>.citations.json`.

This map is the main bridge from low-level evidence to high-level summarisation and layout.

5.8 Engineer-oriented summariser (LLM)

- Modules: `llm/summarize_engineer.py`, `llm/client_vllm.py`.
- Behaviour:
  - assemble a system prompt that frames the model as an engineering-oriented scientific summariser with strict evidence-only constraints;
  - include citation map entries and key spans in the prompt;
  - call the vLLM server via `/v1/completions` using the configured model name and decoding parameters;
  - parse the completion into highlight items, each ending with explicit references.
- Output: `storage/evidence/_rank/<run_id>.summary.json`.

In the captured logs, the long list of token ids and the HTTP 200 status lines are from this stage when Qwen generates the summary.

If the LLM call fails, there is an extractive fallback that selects sentences from the top-ranked evidence spans so that a degraded but still grounded report can be produced.

5.9 Chart generation and rendering

- LLM side: `llm/chart_codegen.py`.
- Renderer: `viz/chart_builder.py`.
- Behaviour:
  - inspect tables in each evidence pack and construct compact text descriptions (column names, ranges, small samples);
  - ask the LLM to emit a Matplotlib function called `render_chart(csv_path, out_path)`;
  - statically validate generated code against a strict import whitelist;
  - execute the code in a sandbox to render PNG charts;
  - deduplicate charts using a stable hash and rank them by usefulness.
- Output:
  - chart PNG files and scripts in `storage/renders/<doc_id>/charts/`;
  - chart metadata in `_charts.json`.

These generated charts appear alongside extracted figures in the final reports.

5.10 HTML report and PPTX export

- HTML builder: `report/layout.py`.
- PPTX exporter: `report/export_pptx.py`.

HTML report

- reads citation map, summary, evidence packs and chart metadata;
- copies all referenced images into `storage/reports/_runs/<run_id>_assets/`;
- constructs a two-column layout where:
  - the left column shows charts first, then web images, then PDF figures;
  - the right column contains grouped bullet highlights with their citations.

The resulting file `storage/reports/<run_id>.html` is what was opened in Jupyter during testing.

PPTX export

- creates a title slide containing the query and run identifier;
- generates one slide per important visual, with a caption area containing highlight text and references;
- appends reference slides summarising the cited papers.

The deck is stored at `storage/reports/<run_id>.pptx` and mirrors the HTML narrative.

5.11 QA checks and trace manifest

- Modules: `qa/consistency_checks.py`, `qa/trace_manifest.py`.
- Behaviour:
  - run basic checks on packs, downloads and reports (for example non-empty packs, presence of expected files);
  - record warnings but avoid aborting the run;
  - build a manifest that summarises retrieval counts, document coverage, figure and chart counts, and QA results.
- Output: `storage/reports/_runs/<run_id>.manifest.json`.

Log messages such as “qa.checks” and “trace.manifest” correspond to this stage.

5.12 Preview server (optional)

- Module: `review/preview_server.py`.
- Role:
  - provide a minimal HTTP interface for browsing runs and artefacts under `storage/`;
  - intended for local developer review; not hardened for public deployment.

# 6. Command-Line Usage

Typical end-to-end run on a Unix-like system:

1. Activate virtual environment

   source .venv/bin/activate

2. Configure local vLLM endpoint and core flags

   export OPENAI_API_BASE=http://127.0.0.1:8000/v1  
   export OPENAI_API_KEY=local-123  
   export CITEVIZOR_DISABLE_GUIDED=1  
   export SUMMARIZER_JSON_STRICT=0  
   (plus SERPER_API_KEY and other secrets loaded from a local env file)

3. Launch one CiteVizor run

   python citevizor_backend.py \
     --query "Dynamic Chain-of-Thought: Towards Adaptive Deep Reasoning" \
     --lang en \
     --k-scholar 12 \
     --k-web 6 \
     --topk-rank 80 \
     --limit-tables 6 \
     --enable-ocr \
     --export-pptx \
     --years ">=2025"

On success, the script prints something like:

   run_id : 98da1d971009  
   HTML   : storage/reports/98da1d971009.html  
   PPTX   : storage/reports/98da1d971009.pptx  
   Manifest: storage/reports/_runs/98da1d971009.manifest.json  

These paths match the HTML and PPTX artefacts inspected during current testing.

# 7. Honest Status Summary

- The backend already:
  - connects Scholar / web search, PDF parsing, evidence curation, LLM reasoning and visual generation into a working end-to-end chain;
  - produces reproducible HTML and PPTX reports with explicit citations and a strong emphasis on images and charts;
  - logs each stage and keeps all artefacts on disk for later debugging and research.

- The backend still:
  - lacks a polished front-end, user management and infrastructure integration;
  - relies on limited manual testing instead of a comprehensive benchmark;
  - may fail or degrade on corner cases outside the currently tested paper set.

In its present form, CiteVizor should be seen as a **technical foundation for an academic assistant**, not as a finished commercial product: it demonstrates the core ideas clearly and honestly, but still expects future work on evaluation, robustness and productisation.
