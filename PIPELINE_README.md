# Local retrieve-and-cite legal RAG

Retrieval is fully local (sentence-transformers + Chroma + BM25). The only
network call is the final answer generation via the Claude API.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python main.py index          # build the local index (downloads MiniLM once)
.venv/bin/python main.py eval           # score retrieval against answer_key.json
export ANTHROPIC_API_KEY=sk-ant-...     # needed only for generation
.venv/bin/python main.py ask "Which parties relinquished control of their shares?" -k 8
.venv/bin/python main.py ask "What severance applies?" --party "Anita Desai"
.venv/bin/python main.py ask "..." --retrieve-only --show-chunks   # no API call
```

## Stages (one file each)

| Stage | File | What it does |
|---|---|---|
| load | `load.py` | Reads every `.txt` in `legal_corpus/`; filename becomes metadata. Skips `README.txt` — it spells out the eval answers and would contaminate retrieval. |
| chunk | `chunk.py` | Paragraph-pack chunking; parses party names from defined terms (`Name (the "Indemnitee")`) with a `between X and Y` recital fallback. Every chunk carries `source`, `party`, `parties`. |
| embed | `embed.py` | all-MiniLM-L6-v2 embeddings into persistent Chroma (`index_store/`, telemetry off). The embedded text is prefixed `[source \| party: ...]` so near-identical chunks separate in vector space; the stored text stays verbatim for quoting. |
| retrieve | `retrieve.py` | Hybrid: dense cosine + BM25 (party and filename tokens included), fused with reciprocal-rank fusion. `--party` resolves a name to its documents and constrains both retrievers. |
| generate | `generate.py` | Claude API (`claude-opus-4-8`, override with `RAG_CLAUDE_MODEL`). Chunks are sent as document blocks with **API-native citations enabled**, so quotes are machine-verified spans of the sources; the system prompt forbids outside knowledge and requires per-claim file citations. |
| eval | `eval.py` | Runs the README's suggested queries; expected document sets are derived from `answer_key.json` (type/party/carve-out flags). Reports PASS/FAIL per query, decoy placement, and audits extracted party metadata against the key. Exit code 0 only if everything passes. |

`main.py` wires them into `index` / `ask` / `eval` subcommands; `config.py`
holds paths and tunables.

## Why party metadata is load-bearing

Five indemnification amendments differ only by party name and one clause.
Semantic search alone sees them as near-duplicates. Three things make the
pipeline distinguish them: the party header baked into each embedding, party
tokens in the BM25 index, and the `--party` metadata filter. The eval's Q2
("Find the change-in-control carve-out for David Okafor") asserts the right
amendment ranks #1 among its four look-alikes.
