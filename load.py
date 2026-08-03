"""Stage 1 — load: read every .txt in the corpus folder.

Each document becomes {"source": <filename>, "text": <raw text>}. The filename
is the only metadata captured here; party extraction happens in chunk.py where
the text is parsed.
"""

from pathlib import Path

import config


def load_documents(corpus_dir: Path = config.CORPUS_DIR) -> list[dict]:
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {corpus_dir}")

    docs = []
    for path in sorted(corpus_dir.glob("*.txt")):
        if path.name in config.EXCLUDE_FILES:
            continue
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            continue
        docs.append({"source": path.name, "text": text})

    if not docs:
        raise FileNotFoundError(f"no .txt documents found in {corpus_dir}")
    return docs


if __name__ == "__main__":
    documents = load_documents()
    print(f"Loaded {len(documents)} documents from {config.CORPUS_DIR}:")
    for doc in documents:
        print(f"  {doc['source']:45s} {len(doc['text']):>6,} chars")
