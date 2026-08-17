# Scale-Corpus Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> *Session note:* this plan is executed inline in the authoring session (executing-plans style); code detail below is given where design is non-obvious, and the spec (`docs/superpowers/specs/2026-08-17-scale-corpus-hardening-design.md`) is the authority on intent.

**Goal:** Make the legal RAG pipeline work end-to-end on a 500+-real-contract corpus with section-aware chunking, batched indexing, and a two-part eval (needle precision + scale recall/latency) that passes.

**Architecture:** Keep the six single-file stages. Add `fetch_corpus.py` (CUAD acquisition) and `tests/`. Chunking becomes section-aware; indexing becomes batched with a manifest + BM25 cache; eval gains a CUAD-derived scale section.

**Tech Stack:** Python 3.13, sentence-transformers (all-MiniLM-L6-v2), ChromaDB, rank_bm25, anthropic SDK, pytest.

## Global Constraints

- Chunk sizes must keep `[header] + body` within MiniLM's 256-wordpiece window: target ≈700 chars, hard max 1500.
- `legal_corpus/*.txt` (synthetic needle set) is committed; `legal_corpus/cuad/` and `data/` are gitignored.
- Retrieval stages stay fully local; only `generate.py` touches the network.
- Filenames and party names of the needle set must match `answer_key.json` exactly.
- Real measured numbers only in docs — never invented figures.
- Commit after each green task; all commits on `harden/scale-corpus`.

---

### Task 1: Reconstruct the synthetic needle corpus

**Files:**
- Create: `legal_corpus/indemnification_{margaret_ellison,david_okafor,priya_raman,thomas_vance,susan_cho}.txt`
- Create: `legal_corpus/employment_{robert_iglesias,anita_desai}.txt`
- Create: `legal_corpus/merger_agreement_coastal.txt`, `legal_corpus/mutual_nda_vantage.txt`, `legal_corpus/commercial_lease_harborpoint.txt`
- Create: `legal_corpus/README.txt` (copy of root README.txt; excluded from indexing by config)
- Modify: `.gitignore` (ignore only `legal_corpus/cuad/`, not all of `legal_corpus/`)

**Interfaces:**
- Produces: 10 docs whose defined terms parse under `chunk.extract_parties` (`Name (the "Role")` drafting; NDA uses only a `between X and Y` recital), and whose content satisfies every case in `eval.build_cases`.

**Content requirements (from answer_key.json + README.txt):**
- 5 indemnification amendments, near-identical, differing only by party and one clause. Ellison/Okafor/Vance: change-in-control carve-out where the Indemnitee **relinquished control of the shares**. Raman/Cho: the same section explicitly **negates** the carve-out ("shall NOT be deemed to have relinquished control…"). All five reference the Coastal Devices merger agreement by date (Effective Time).
- 2 employment agreements (Iglesias, Desai) with change-of-control severance provisions; Desai's includes severance terms (README: "What severance applies? --party Anita Desai").
- Merger agreement: Coastal Devices stockholders **relinquish control** in the different, transactional sense.
- NDA (Vantage Analytics): confidentiality only, explicitly **no** change-of-control provision; parties introduced only via recital ("between Vantage Analytics … and …"). Decoy.
- Lease (Harbor Point landlord): "change in control" only re: assignment consent. Decoy.

**Steps:**
- [ ] Write the 10 documents + README copy.
- [ ] `.venv/bin/python chunk.py` — verify party extraction matches the key for every doc.
- [ ] `.venv/bin/python main.py index` then `.venv/bin/python main.py eval` — all queries PASS, party metadata ok.
- [ ] Fix documents (not the eval) until green.
- [ ] Update `.gitignore`; `git add legal_corpus/*.txt`; commit `feat: reconstruct synthetic needle corpus (eval green)`.

### Task 2: CUAD corpus fetcher

**Files:**
- Create: `fetch_corpus.py`
- Modify: `.gitignore` (add `data/`)

**Interfaces:**
- Produces: `legal_corpus/cuad/<safe_name>.txt` (510 contracts), `data/cuad_annotations.json` mapping `cuad/<safe_name>.txt` → `{"parties": [...], "provisions": {"governing_law": bool, "change_of_control": bool, ...}}` derived from CUAD QA annotations.
- CLI: `.venv/bin/python fetch_corpus.py [--force]`, idempotent (skips when count matches).

**Design:**
- Candidate sources tried in order (first that works wins), each verified by contract count == 510 and total bytes > 10MB:
  1. GitHub `TheAtticusProject/cuad` repo `data.zip` (CUADv1.json inside).
  2. HuggingFace `datasets/theatticusproject/cuad` resolve URLs.
  3. Zenodo record 4595826 `CUAD_v1.zip` (large; last resort).
- Extract to a temp dir, then atomically move into place. Safe filenames via slugify, collision-checked.
- Slim annotations: for each contract take QA categories `Parties`, `Governing Law`, `Change of Control`, `Anti-assignment`, `Expiration Date` → presence + answer texts.

**Steps:**
- [ ] Implement; run `.venv/bin/python fetch_corpus.py`; verify 510 files + annotations exist.
- [ ] Spot-read 2 contracts for text sanity (headers, sections present).
- [ ] Commit `feat: add CUAD corpus fetcher` (code only; corpus stays untracked).

### Task 3: Section-aware chunking + robust party extraction

**Files:**
- Modify: `load.py` (recursive load), `chunk.py`, `config.py`
- Test: `tests/test_chunk.py`

**Interfaces:**
- Consumes: `load.load_documents() -> [{"source", "text"}]` where source is corpus-relative (e.g. `cuad/FOO.txt`).
- Produces: chunks gain `"section": str` (heading path, may be ""); `chunk_text(text) -> list[dict]` now returns `[{"text", "section"}]`; `chunk_documents` output otherwise unchanged (`id`, `source`, `chunk_index`, `party`, `parties`, `text`).

**Heading detection (line-based, any of):**
```python
_HEADING_RES = [
    re.compile(r"^\s*(ARTICLE|SECTION)\s+[IVXLC0-9]+", re.IGNORECASE),
    re.compile(r"^\s*(?:Section\s+)?\d+(?:\.\d+)*[.)]?\s+[A-Z\"']"),  # 1. / 2.3 Definitions
    re.compile(r"^\s*[A-Z][A-Z &,'/-]{4,60}\.?\s*$"),                  # ALL-CAPS line
]
```
A heading line starts a new section; packing never crosses a section boundary. Section string = most recent heading, whitespace-normalized, ≤80 chars.

**Party extraction additions:** roles extended (`purchaser, seller, buyer, licensor, licensee, distributor, tenant, lessee, lessor, borrower, lender, consultant, contractor, supplier, customer, client, vendor`); recital regex also matches `by and between/among X and Y`; results deduped/cleaned; empty result stays legal.

**Steps:**
- [ ] Write failing tests: heading detection cases, no-merge-across-sections, hard-split invariants (`max ≤ 1500`), party extraction on a real CUAD-style excerpt and on each needle-doc pattern.
- [ ] `.venv/bin/python -m pytest tests/test_chunk.py -q` → fails.
- [ ] Implement; tests pass.
- [ ] Re-run needle `index`+`eval` (10 docs) — still green (chunking changes must not regress it).
- [ ] Commit `feat: section-aware chunking + hardened party extraction`.

### Task 4: Batched indexing, manifest, BM25 cache

**Files:**
- Modify: `embed.py`, `retrieve.py`, `config.py`

**Interfaces:**
- Produces: `index_store/manifest.json` `{embed_model, chunk_target, chunk_max, corpus_fingerprint, n_docs, n_chunks, built_at}`; `index_store/bm25.pkl` (tokenized corpus, pickled); `embed.embedding_input(chunk)` header format `[{stem≤48} | {party|unknown}{ | section}]`.
- `retrieve.Retriever` loads bm25.pkl when manifest fingerprint matches `chunks.jsonl`; hard error if manifest's `embed_model != config.EMBED_MODEL`.

**Design notes:**
- Encode in batches of 256 (`show_progress_bar=True` when >1000 chunks); Chroma adds in batches of `min(client.get_max_batch_size(), 5000)`.
- `corpus_fingerprint` = sha256 over sorted `(source, size, mtime_ns)` of corpus files — cheap, catches drift.
- Manifest written **after** all adds (crash → stale manifest → Retriever rebuilds BM25 and warns).

**Steps:**
- [ ] Implement; `.venv/bin/python main.py index` on needle corpus — verify manifest + bm25.pkl written, eval still green.
- [ ] Run full index with CUAD present; record wall time and chunk count.
- [ ] Commit `feat: batched indexing with manifest and BM25 cache`.

### Task 5: Scale eval + needle-at-scale tuning

**Files:**
- Modify: `eval.py`, `config.py`, `main.py` (`eval --skip-scale` flag)
- Test: `tests/test_retrieve.py` (tokenize + RRF math on toy fixtures)

**Interfaces:**
- Produces: `eval.run_scale_eval(retriever, annotations, n=100, k=5) -> {"recall_at_k", "mrr", "n", "latency_ms": {"p50","p95"}}`; `main` exit 0 requires needle PASS **and** recall@5 ≥ 0.8 when scale corpus present.

**Query synthesis from annotations (deterministic, seeded):** for contracts with a `Parties` annotation whose distinctive party token-set is unique in the corpus, and a present provision, generate e.g. `"What is the governing law of the agreement involving {party}?"` / `"Does the {party} agreement contain a change of control provision?"`. Expected doc = that contract; graded document-level (rank of expected doc in fused doc ranking).

**Steps:**
- [ ] Write toy-fixture tests for tokenize/RRF; pass.
- [ ] Implement scale eval; run full `eval` with CUAD indexed.
- [ ] If needle cases fail at scale: raise `CANDIDATE_DEPTH` (32→64→96), re-check; record final value + numbers.
- [ ] Commit `feat: CUAD-derived scale eval; tune retrieval depth for 40k+ chunks`.

### Task 6: Generation polish

**Files:**
- Modify: `generate.py`, `config.py`

**Steps:**
- [ ] `config.py`: autoload `.env` (stdlib parser: `KEY=VALUE` lines → `os.environ.setdefault`; never log values); default model → `claude-opus-5` (keep `RAG_CLAUDE_MODEL` override).
- [ ] `generate.py`: switch to `client.messages.stream(...)` + `stream.get_final_message()` (long document input; SDK guidance), keep citations rendering and error mapping unchanged.
- [ ] `.venv/bin/python main.py ask "Find the change-in-control carve-out for David Okafor." --retrieve-only` — retrieval path clean; live API call attempted once if credentials resolve (skip gracefully otherwise).
- [ ] Commit `feat: stream generation on claude-opus-5; autoload .env`.

### Task 7: Docs, demo, final verification

**Files:**
- Modify: `PIPELINE_README.md`; Create: `DEMO.md`

**Steps:**
- [ ] Full clean-slate verification: delete `index_store/`, `fetch_corpus.py` (idempotent no-op), `index`, `eval`, pytest — all green.
- [ ] Rewrite PIPELINE_README: quickstart incl. fetcher, stage table updates, measured numbers (docs, chunks, index time, latency, recall, needle results).
- [ ] DEMO.md: 5-minute interview walkthrough (the near-dupe precision story at 500× noise, the numbers, the architecture talking points).
- [ ] Commit `docs: scale numbers, demo walkthrough`; final `git log` review.

## Self-review

Spec coverage: every spec bullet maps to a task (fetcher→2, load/chunk→3, embed/retrieve→4, eval→5, generate→6, docs→7, corpus reconstruction→1). Placeholders: none — content requirements are spelled out where the executor needs them. Type consistency: chunk dict fields (`section`), manifest keys, and `run_scale_eval` signature are used consistently across tasks 3–5.
