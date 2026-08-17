"""Shared configuration for the legal RAG pipeline."""

import os
from pathlib import Path

# Keep local stages local: no Hugging Face telemetry (the model download on
# first index build is the only hub traffic).
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

PROJECT_ROOT = Path(__file__).resolve().parent
CORPUS_DIR = PROJECT_ROOT / "legal_corpus"
INDEX_DIR = PROJECT_ROOT / "index_store"
CHUNKS_FILE = INDEX_DIR / "chunks.jsonl"
CHROMA_DIR = INDEX_DIR / "chroma"
ANSWER_KEY_FILE = PROJECT_ROOT / "answer_key.json"

# legal_corpus/ contains a copy of README.txt, which describes the test setup
# and literally spells out the expected answers. Indexing it would contaminate
# retrieval (and the eval), so it is excluded. answer_key.json is not a .txt,
# so the loader never sees it.
EXCLUDE_FILES = {"README.txt"}

# Embeddings are computed locally; only generate.py talks to the network.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLAUDE_MODEL = os.environ.get("RAG_CLAUDE_MODEL", "claude-opus-4-8")

COLLECTION_NAME = "legal_chunks"

# Chunking. Sizes are governed by the embedder: all-MiniLM-L6-v2 truncates at
# 256 wordpieces (~900-1200 chars), so the [header] + body of a chunk must fit
# that window — bigger chunks would be silently cut off in embedding space
# (BM25 still sees the full text).
CHUNK_TARGET_CHARS = 700  # pack adjacent pieces up to roughly this size
CHUNK_MAX_CHARS = 1200  # split anything longer, at sentence boundaries
CHUNK_OVERLAP_CHARS = 150  # overlap used only for no-sentence-break splits

# Retrieval
CANDIDATE_DEPTH = 32  # ranked-list depth per retriever before fusion
RRF_K = 60  # standard reciprocal-rank-fusion constant
DEFAULT_TOP_K_CHUNKS = 6  # chunks handed to the generator
DEFAULT_TOP_K_DOCS = 5  # document-level cutoff used by the eval
