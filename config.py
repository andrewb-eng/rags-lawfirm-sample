"""Shared configuration for the legal RAG pipeline."""

import os
from pathlib import Path

# Keep local stages local: no Hugging Face telemetry (the model download on
# first index build is the only hub traffic).
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader: KEY=VALUE lines feed os.environ defaults.

    Real environment variables always win, values are never logged, and the
    file is optional — this only saves the `export ANTHROPIC_API_KEY=...`
    step before `python main.py ask`.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().removeprefix("export ")
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)


_load_dotenv(Path(__file__).resolve().parent / ".env")

PROJECT_ROOT = Path(__file__).resolve().parent
CORPUS_DIR = PROJECT_ROOT / "legal_corpus"
INDEX_DIR = PROJECT_ROOT / "index_store"
CHUNKS_FILE = INDEX_DIR / "chunks.jsonl"
CHROMA_DIR = INDEX_DIR / "chroma"
MANIFEST_FILE = INDEX_DIR / "manifest.json"  # written last; absence = stale build
BM25_CACHE_FILE = INDEX_DIR / "bm25.pkl"
ANSWER_KEY_FILE = PROJECT_ROOT / "answer_key.json"
CUAD_ANNOTATIONS_FILE = PROJECT_ROOT / "data" / "cuad_annotations.json"

# legal_corpus/ contains a copy of README.txt, which describes the test setup
# and literally spells out the expected answers. Indexing it would contaminate
# retrieval (and the eval), so it is excluded. answer_key.json is not a .txt,
# so the loader never sees it.
EXCLUDE_FILES = {"README.txt"}

# Embeddings are computed locally; only generate.py talks to the network.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLAUDE_MODEL = os.environ.get("RAG_CLAUDE_MODEL", "claude-opus-5")

COLLECTION_NAME = "legal_chunks"

# Chunking. Sizes are governed by the embedder: all-MiniLM-L6-v2 truncates at
# 256 wordpieces (~900-1200 chars), so the [header] + body of a chunk must fit
# that window — bigger chunks would be silently cut off in embedding space
# (BM25 still sees the full text).
CHUNK_TARGET_CHARS = 700  # pack adjacent pieces up to roughly this size
CHUNK_MAX_CHARS = 1200  # split anything longer, at sentence boundaries
CHUNK_OVERLAP_CHARS = 150  # overlap used only for no-sentence-break splits

# Indexing at scale
EMBED_BATCH_SIZE = 128  # sentence-transformers encode batch
CHROMA_ADD_BATCH = 5000  # ceiling per collection.add (Chroma rejects huge adds)

# Retrieval
CANDIDATE_DEPTH = 64  # ranked-list depth per retriever before fusion
DOC_CANDIDATE_DEPTH = 200  # deeper pool for document-level ranking: near-dupe
# documents interleave many chunks at the top, so doc rankings need to see
# past them (see Retriever.retrieve_docs)
HYBRID_ALPHA = 0.5  # dense weight in relative-score fusion (1-alpha = BM25);
# see retrieve.py's docstring for why score fusion replaced rank fusion
DEFAULT_TOP_K_CHUNKS = 6  # chunks handed to the generator
DEFAULT_TOP_K_DOCS = 5  # document-level cutoff used by the eval

# Scale eval (CUAD-derived; runs when the bulk corpus is indexed)
SCALE_EVAL_N = 100  # queries synthesized from contract annotations
SCALE_RECALL_TARGET = 0.8  # recall@DEFAULT_TOP_K_DOCS required to pass
