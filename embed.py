"""Stage 3 — embed: encode chunks with sentence-transformers into local Chroma.

Everything here runs on-machine: the MiniLM model is downloaded once from the
Hugging Face hub and inference is local; Chroma persists to ./index_store with
telemetry disabled. The raw chunk text is stored as the Chroma document so
generation can quote it verbatim, while the *embedded* text is prefixed with a
`[source | party | section]` header — that header is what separates
near-identical chunks in vector space when their bodies differ by only a name.

Scale hardening:
  * encoding runs in batches with a progress bar on large corpora
  * Chroma adds are batched under the client's max batch size (a single
    50k-item add is rejected outright)
  * a manifest (model, chunk config, corpus fingerprint, counts) is written
    LAST, so a crashed build is detectably incomplete and retrieval can
    refuse politely instead of serving a half-index
"""

import json
import time

import chromadb
from chromadb.config import Settings

import config
from chunk import chunk_documents
from load import corpus_fingerprint, load_documents

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(config.EMBED_MODEL)
    return _embedder


def embed_texts(texts: list[str], progress: bool = False) -> list[list[float]]:
    vectors = get_embedder().encode(
        texts,
        batch_size=config.EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=progress,
    )
    return vectors.tolist()


def embedding_input(chunk: dict) -> str:
    """Header + body actually fed to the embedder.

    The header disambiguates near-identical chunks (party) and situates the
    text (document stem, section). It is capped so long CUAD filenames don't
    eat into MiniLM's 256-wordpiece window; the stored text stays verbatim
    for quoting.
    """
    stem = chunk["source"].rsplit("/", 1)[-1].removesuffix(".txt")[:48]
    party = chunk["party"] or "unknown"
    section = chunk.get("section", "")
    header = f"[{stem} | party: {party}"
    if section:
        header += f" | {section[:60]}"
    return f"{header}] {chunk['text']}"


def get_chroma_client() -> chromadb.ClientAPI:
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(config.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection():
    return get_chroma_client().get_collection(config.COLLECTION_NAME)


def build_index(chunks: list[dict], n_docs: int, fingerprint: str) -> int:
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    # A rebuild invalidates everything derived from the old index.
    config.MANIFEST_FILE.unlink(missing_ok=True)
    config.BM25_CACHE_FILE.unlink(missing_ok=True)

    # chunks.jsonl is the manifest of record for chunk text/metadata:
    # retrieve.py builds BM25 from it and the eval reads party metadata from
    # it, so it always mirrors the vector store.
    with config.CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")

    client = get_chroma_client()
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    show_progress = len(chunks) > 2000
    vectors = embed_texts(
        [embedding_input(c) for c in chunks], progress=show_progress
    )

    try:
        max_batch = min(client.get_max_batch_size(), config.CHROMA_ADD_BATCH)
    except Exception:
        max_batch = config.CHROMA_ADD_BATCH
    for start in range(0, len(chunks), max_batch):
        batch = chunks[start : start + max_batch]
        collection.add(
            ids=[c["id"] for c in batch],
            embeddings=vectors[start : start + max_batch],
            documents=[c["text"] for c in batch],
            # Chroma metadata values must be scalars, so the party list is JSON.
            metadatas=[
                {
                    "source": c["source"],
                    "chunk_index": c["chunk_index"],
                    "party": c["party"],
                    "parties": json.dumps(c["parties"]),
                    "section": c.get("section", ""),
                }
                for c in batch
            ],
        )

    count = collection.count()
    config.MANIFEST_FILE.write_text(
        json.dumps(
            {
                "embed_model": config.EMBED_MODEL,
                "chunk_target_chars": config.CHUNK_TARGET_CHARS,
                "chunk_max_chars": config.CHUNK_MAX_CHARS,
                "corpus_fingerprint": fingerprint,
                "n_docs": n_docs,
                "n_chunks": count,
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    return count


def read_manifest() -> dict | None:
    if not config.MANIFEST_FILE.exists():
        return None
    return json.loads(config.MANIFEST_FILE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    documents = load_documents()
    all_chunks = chunk_documents(documents)
    n = build_index(all_chunks, len(documents), corpus_fingerprint())
    print(f"Indexed {n} chunks into {config.CHROMA_DIR} "
          f"(collection '{config.COLLECTION_NAME}', model {config.EMBED_MODEL})")
