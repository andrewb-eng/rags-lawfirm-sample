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

# Chunking: documents in this corpus are ~1 KB, so chunks are paragraph packs.
CHUNK_TARGET_CHARS = 600  # merge adjacent paragraphs up to roughly this size
CHUNK_MAX_CHARS = 1500  # hard-split anything longer than this
CHUNK_OVERLAP_CHARS = 150  # overlap used only for hard splits

# Retrieval
CANDIDATE_DEPTH = 32  # ranked-list depth per retriever before fusion
RRF_K = 60  # standard reciprocal-rank-fusion constant
DEFAULT_TOP_K_CHUNKS = 6  # chunks handed to the generator
DEFAULT_TOP_K_DOCS = 5  # document-level cutoff used by the eval
