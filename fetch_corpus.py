"""Fetch the CUAD v1 bulk corpus: 510 real commercial contracts.

CUAD (Contract Understanding Atticus Dataset, CC-BY-4.0) ships as a single
SQuAD-style JSON containing every contract's full text plus expert
annotations. This script extracts:

  legal_corpus/cuad/<slug>.txt     one file per contract (indexed by load.py)
  data/cuad_annotations.json       slim per-contract ground truth used by the
                                   scale eval (parties + provision presence)

Both locations are gitignored — the corpus is rebuildable from this script.
Idempotent: re-running with a complete corpus is a no-op (use --force to
redo). The downloaded archive is cached in data/ so retries don't re-fetch.
"""

import argparse
import io
import json
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import config

CUAD_DIR = config.CORPUS_DIR / "cuad"
DATA_DIR = config.PROJECT_ROOT / "data"
ZIP_CACHE = DATA_DIR / "cuad_data.zip"
ANNOTATIONS_FILE = DATA_DIR / "cuad_annotations.json"

EXPECTED_CONTRACTS = 510
MIN_TOTAL_BYTES = 10_000_000

# (url, human-readable size) tried in order; each must contain CUADv1.json.
SOURCES = [
    ("https://github.com/TheAtticusProject/cuad/raw/main/data.zip", "~17 MB"),
    ("https://zenodo.org/record/4595826/files/CUAD_v1.zip", "~433 MB"),
]

# Annotation categories kept for the scale eval. CUAD question ids end in
# "__<Category>"; answers are verbatim spans of the contract text.
KEPT_CATEGORIES = [
    "Document Name",
    "Parties",
    "Agreement Date",
    "Expiration Date",
    "Governing Law",
    "Change Of Control",
    "Anti-Assignment",
]


def _download(dest: Path) -> None:
    import requests  # transitive dependency of sentence-transformers

    last_err = None
    for url, size in SOURCES:
        print(f"Downloading CUAD ({size}) from {url}")
        try:
            with requests.get(url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                done = 0
                with dest.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        print(f"\r  {done / 1e6:,.0f} MB", end="", flush=True)
            print()
            return
        except Exception as err:  # try the next mirror
            last_err = err
            print(f"  failed: {err}", file=sys.stderr)
            dest.unlink(missing_ok=True)
    raise SystemExit(
        f"Could not download CUAD from any source (last error: {last_err}). "
        "Check your network, or download data.zip manually from "
        f"github.com/TheAtticusProject/cuad into {ZIP_CACHE}."
    )


def _load_cuad_json(zip_path: Path) -> dict:
    with zipfile.ZipFile(zip_path) as zf:
        member = next(
            (n for n in zf.namelist() if n.endswith("CUADv1.json")), None
        )
        if member is None:
            raise SystemExit(f"{zip_path} does not contain CUADv1.json")
        with zf.open(member) as f:
            return json.load(io.TextIOWrapper(f, encoding="utf-8"))


def _slugify(title: str, seen: set[str]) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")[:120].lower()
    base, n = slug, 2
    while slug in seen:
        slug, n = f"{base}_{n}", n + 1
    seen.add(slug)
    return slug


def _annotate(qas: list[dict]) -> dict:
    """Reduce a contract's QA list to {category: {present, answers}}."""
    by_cat = {}
    for qa in qas:
        cat = qa["id"].rsplit("__", 1)[-1]
        if cat not in KEPT_CATEGORIES:
            continue
        answers = [
            " ".join(a["text"].split())[:300] for a in qa["answers"][:3]
        ]
        by_cat[cat] = {"present": not qa["is_impossible"], "answers": answers}
    return by_cat


def corpus_is_complete() -> bool:
    return (
        ANNOTATIONS_FILE.exists()
        and CUAD_DIR.is_dir()
        and len(list(CUAD_DIR.glob("*.txt"))) == EXPECTED_CONTRACTS
    )


def main(force: bool = False) -> int:
    if corpus_is_complete() and not force:
        print(
            f"CUAD corpus already present ({EXPECTED_CONTRACTS} contracts in "
            f"{CUAD_DIR}); nothing to do. Use --force to refetch."
        )
        return 0

    DATA_DIR.mkdir(exist_ok=True)
    if not ZIP_CACHE.exists():
        _download(ZIP_CACHE)
    else:
        print(f"Using cached archive {ZIP_CACHE}")

    cuad = _load_cuad_json(ZIP_CACHE)
    contracts = cuad["data"]
    if len(contracts) != EXPECTED_CONTRACTS:
        raise SystemExit(
            f"Expected {EXPECTED_CONTRACTS} contracts, found {len(contracts)}"
        )

    # Extract into a temp dir, then move into place so a crash can't leave a
    # half-written corpus behind.
    seen: set[str] = set()
    annotations: dict[str, dict] = {}
    total_bytes = 0
    tmp = Path(tempfile.mkdtemp(prefix="cuad_", dir=config.PROJECT_ROOT))
    try:
        for entry in contracts:
            paragraph = entry["paragraphs"][0]
            slug = _slugify(entry["title"], seen)
            text = paragraph["context"]
            (tmp / f"{slug}.txt").write_text(text, encoding="utf-8")
            total_bytes += len(text.encode("utf-8"))
            annotations[f"cuad/{slug}.txt"] = {
                "title": entry["title"],
                **{"categories": _annotate(paragraph["qas"])},
            }

        if total_bytes < MIN_TOTAL_BYTES:
            raise SystemExit(
                f"Extracted corpus suspiciously small ({total_bytes:,} bytes)"
            )

        if CUAD_DIR.exists():
            shutil.rmtree(CUAD_DIR)
        tmp.rename(CUAD_DIR)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)

    ANNOTATIONS_FILE.write_text(
        json.dumps(annotations, indent=1), encoding="utf-8"
    )
    print(
        f"Wrote {len(contracts)} contracts ({total_bytes / 1e6:,.1f} MB) to "
        f"{CUAD_DIR}\nWrote annotations for the scale eval to {ANNOTATIONS_FILE}"
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--force", action="store_true", help="refetch even if corpus complete"
    )
    sys.exit(main(force=parser.parse_args().force))
