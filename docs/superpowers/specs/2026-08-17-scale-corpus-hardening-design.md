# Scale-corpus hardening — design

**Date:** 2026-08-17 · **Branch:** `harden/scale-corpus` · **Status:** approved-by-default (autonomous session; see Assumptions)

## Goal

Harden the legal RAG pipeline so it works on a genuinely large corpus — properly
chunked, embedded, and measured — producing a demo that is credible on a resume:
real numbers, real documents, and an adversarial eval that still passes at scale.

## Current state

Six single-file stages (`load → chunk → embed → retrieve → generate → eval`)
built around a 10-document synthetic corpus of near-identical legal agreements
(the "needle set"), with party-metadata-aware chunking, hybrid retrieval
(MiniLM dense + BM25, RRF-fused), Claude generation with API-native citations,
and a strict retrieval eval derived from `answer_key.json`.

**Blocking defect:** `legal_corpus/` was gitignored wholesale and did not
survive the machine migration — the corpus is gone and the pipeline cannot run.
The synthetic docs must be reconstructed from their specification
(`answer_key.json` + `README.txt`), and the eval is the objective check that the
reconstruction is faithful.

## Assumptions (made autonomously — flag if wrong)

1. **Corpus choice is delegated** ("give it a really long corpus"). I chose
   CUAD v1 (see below).
2. The 6-stage single-file architecture is the project's identity; hardening
   extends it rather than replacing it with a framework.
3. The synthetic needle set becomes committed fixture data (tiny, load-bearing
   for the eval); only the bulk corpus stays gitignored.
4. Local-first retrieval stays (sentence-transformers + Chroma + BM25); the only
   network stage remains generation via the Claude API.

## Corpus: approaches considered

| Option | Pros | Cons |
|---|---|---|
| **A. CUAD v1 (chosen)** — 510 real commercial contracts (Atticus Project, CC-BY-4.0), ~30MB of text, with expert annotations per contract | Thematically identical to the existing corpus (contracts, parties, change-of-control!), so party extraction and the needle eval extend naturally; annotations give a free ground-truth **scale eval**; real-world, citable dataset | Download source needs fallbacks; contract OCR text is messy (a feature: it exercises chunk hardening) |
| B. CourtListener SCOTUS opinions | Very large, clean | Wrong document genre for a party/contract-oriented pipeline; party regex and eval don't transfer |
| C. Pile-of-law (HF) | Huge | Heavy tooling dependency, no per-doc ground truth, overkill |

**The demo story A enables:** the 5 near-identical indemnification agreements
now hide among 500+ real contracts (~4,000× more text), and retrieval still
ranks David Okafor's amendment #1 — precision that survives 500× noise, plus a
quantitative recall@k over CUAD-annotation-derived queries.

## Chunking: approaches considered

| Option | Verdict |
|---|---|
| Keep paragraph-packing, just retune sizes | Insufficient — 50–200KB contracts need structure awareness for quotable, on-topic chunks |
| **Section-aware paragraph packing (chosen)** — detect legal headings (`ARTICLE IV`, `Section 2.3`, `1. Definitions`, ALL-CAPS lines), never merge across section boundaries, carry a `section` field on each chunk | Structure-aware without new dependencies; heading text also sharpens the embedding header |
| Embedding-drift "semantic chunking" | Expensive at 40k+ chunks, unneeded for legal text which is explicitly sectioned |

**Size constraint that governs everything:** `all-MiniLM-L6-v2` truncates at
256 wordpieces (≈ 900–1200 chars). Chunk target stays ≈700 chars / max 1500 so
the `[header] + body` embedding input fits the window. Bigger chunks would be
silently truncated — the config comments must say this.

## Design by stage

- **`fetch_corpus.py` (new):** downloads CUAD v1, extracts each contract's full
  text to `legal_corpus/cuad/<doc>.txt` and a slim `data/cuad_annotations.json`
  (parties + a few provision annotations per doc) for the scale eval.
  Idempotent; multiple candidate URLs; size/count sanity checks. `data/` and
  `legal_corpus/cuad/` gitignored; synthetic `legal_corpus/*.txt` committed.
- **`load.py`:** recursive `rglob("*.txt")`; `source` becomes the
  corpus-relative path (`cuad/FOO.txt`); same exclusions.
- **`chunk.py`:** heading detection + section-scoped packing as above; party
  extraction gains real-contract patterns (`by and between X and Y`, more roles:
  purchaser, seller, licensor, licensee, distributor, tenant…) and must degrade
  gracefully (empty party is legal). Unit tests in `tests/`.
- **`embed.py`:** encode in batches (progress bar on); add to Chroma in batches
  ≤ the client's max batch size; embedding header becomes
  `[<stem≤48> | <party or unknown> | <section?>]` so long CUAD filenames don't
  eat the window; write `index_store/manifest.json` (model, chunk config,
  corpus fingerprint, counts) and a pickled BM25 cache keyed to the manifest.
- **`retrieve.py`:** load BM25 from cache when fresh (rebuild otherwise);
  manifest/model mismatch fails loudly; `CANDIDATE_DEPTH` becomes scale-tuned
  (raise until the needle eval passes with the full corpus).
- **`generate.py`:** default model `claude-opus-5` (current API default;
  `RAG_CLAUDE_MODEL` still overrides), streaming with `get_final_message()`
  (long document input), `.env` autoloaded by `config.py` via a tiny stdlib
  parser (values never printed).
- **`eval.py`:** unchanged needle eval (must stay green at full scale) + new
  scale section: ~100 queries synthesized from CUAD annotations with a known
  source contract (e.g. governing-law/party queries targeting contracts whose
  party names are unique in the corpus), reporting recall@5, MRR, and retrieval
  latency percentiles. Exit code covers both sections.

## Error handling

Fetcher: fail with actionable messages (which URL failed, what to do), never
leave a half-extracted corpus (write to temp, move into place). Indexing:
batches make partial progress visible; manifest written last so a crashed index
is detectably stale. Retrieval: missing/stale index → clear "run `python
main.py index`" error, not a traceback.

## Testing

- `tests/test_chunk.py`: heading detection, section-scoped packing, party
  extraction on synthetic + real-contract excerpts, size invariants.
- `tests/test_retrieve.py`: tokenizer, RRF fusion math on toy fixtures.
- `python main.py eval` is the integration gate — needle eval green at 10 docs
  **and** at 510+10 docs, plus scale-eval thresholds (recall@5 ≥ 0.8 target).
- Real measured numbers (chunk counts, index time, latency, recall) go in the
  README — no invented figures.

## Out of scope (YAGNI)

Web UI, rerankers/cross-encoders, incremental re-indexing, multi-model embed
comparison, deployment. Mentioned in README as future work at most.
