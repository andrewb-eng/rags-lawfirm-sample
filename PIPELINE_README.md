# Local retrieve-and-cite legal RAG

Hybrid retrieval over **520 legal documents / 50,397 chunks** — 510 real
commercial contracts (CUAD v1) plus a 10-document adversarial "needle set" —
fully local (sentence-transformers + Chroma + BM25). The only network call is
the final answer generation via the Claude API, with API-native citations so
every quoted span is machine-verified against its source document.

## Measured results (Apple Silicon, this repo)

| Metric | Value |
|---|---|
| Corpus | 520 docs, ~27 MB text (510 CUAD contracts + 10 needle docs) |
| Chunks | 50,397 (median 547 chars, max 1,200; 92% carry a section label) |
| Full index build | 81 s end-to-end (2.5 s chunking, ~56 s embedding on MPS) |
| Needle eval | **5/5** adversarial queries pass at 500× noise |
| Scale eval (100 CUAD-annotation queries) | **recall@5 0.92, MRR 0.85** |
| Retrieval latency | p50 88 ms, p95 111 ms per query |

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python fetch_corpus.py         # optional: 510 real contracts (CUAD, ~17 MB download)
.venv/bin/python main.py index           # build the local index (downloads MiniLM once)
.venv/bin/python main.py eval            # needle eval + CUAD scale eval
.venv/bin/python main.py ask "Which parties relinquished control of their shares?"
.venv/bin/python main.py ask "What severance applies?" --party "Anita Desai"
.venv/bin/python main.py ask "..." --retrieve-only --show-chunks   # no API call
```

Generation needs Anthropic credentials (`ANTHROPIC_API_KEY` in the
environment or a local `.env`); retrieval and the eval run fully offline.

## Stages (one file each)

| Stage | File | What it does |
|---|---|---|
| fetch | `fetch_corpus.py` | Downloads CUAD v1 (510 real contracts, CC-BY-4.0), extracts one `.txt` per contract plus a slim annotations file that seeds the scale eval. Idempotent, mirrored sources, atomic extraction. |
| load | `load.py` | Reads every `.txt` under `legal_corpus/` recursively; the corpus-relative path becomes `source`. Skips `README.txt` — it spells out the eval answers and would contaminate retrieval. Also fingerprints the corpus so a stale index warns. |
| chunk | `chunk.py` | Section-aware chunking built for PDF-extracted text: detects legal headings both at line starts and inline after collapsed newlines, never packs across section boundaries, splits over-long pieces at sentence ends. Party extraction uses role-allowlisted defined terms, quoted short-name aliases, and recital parsing that survives address blobs and ALL-CAPS names. Every chunk carries `source`, `party`, `parties`, `section`. |
| embed | `embed.py` | all-MiniLM-L6-v2 embeddings into persistent Chroma, batched end to end. The embedded text is prefixed `[stem \| party \| section]` so near-identical chunks separate in vector space; the stored text stays verbatim for quoting. Chunk sizes are capped to MiniLM's 256-token window. A manifest (model, config, corpus fingerprint) is written last, so crashed builds are detectable. |
| retrieve | `retrieve.py` | Hybrid: dense cosine + BM25 (party and filename tokens included), combined by **relative-score fusion** (per-query min-max normalization, `HYBRID_ALPHA` convex mix). BM25 loads from a pickled cache keyed to the manifest. `--party` resolves names by token containment and constrains both retrievers. Document ranking dedupes a 200-deep fused pool. |
| generate | `generate.py` | Claude API (`claude-opus-5`, override with `RAG_CLAUDE_MODEL`), streamed. Chunks are sent as document blocks with **API-native citations enabled**, so quotes are machine-verified spans of the sources; the system prompt forbids outside knowledge and requires per-claim file citations. |
| eval | `eval.py` | Two sections. **Needle:** the README's adversarial queries, expected sets derived from `answer_key.json`, must pass with the full corpus indexed. **Scale:** ~100 queries synthesized from CUAD's expert annotations (each names a corpus-unique party and targets an annotated provision, so exactly one document is correct); reports recall@5, MRR, latency percentiles. Exit code 0 only if both pass. |

`main.py` wires them into `index` / `ask` / `eval` subcommands; `config.py`
holds paths and tunables.

## Why party metadata is load-bearing

Five indemnification amendments differ only by party name and one clause.
Semantic search alone sees them as near-duplicates — measured here: their
embeddings spread across dense ranks 2–56 for the flagship query, with real
contracts interleaved between them. Three things make the pipeline
distinguish them: the party header baked into each embedding, party tokens in
the BM25 index, and the `--party` metadata filter. The eval's Q2 ("Find the
change-in-control carve-out for David Okafor") asserts the right amendment
ranks #1 among its four look-alikes **and** 510 real contracts.

## Why score fusion replaced rank fusion

This repo originally fused retrievers with reciprocal-rank fusion. At 50k
chunks that failed measurably: on the flagship query the five amendments'
BM25 scores sat 50%+ above every real contract (25.3–25.5 vs ≤17.3), but RRF
compressed that cliff into adjacent ranks, and chunks that were merely
mediocre in both retrievers outranked two of the five. Relative-score fusion
(min-max normalization per query, then `α·dense + (1−α)·BM25`) preserves the
margin. Switching lifted the needle eval from 3/5 to 5/5 **and** the CUAD
scale eval from 0.86 to 0.92 recall@5.

## Future work

Cross-encoder reranking over the fused pool, incremental re-indexing, and a
generation-quality eval (the current scale eval grades retrieval only).
