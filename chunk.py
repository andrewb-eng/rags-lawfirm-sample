"""Stage 2 — chunk: split documents and attach metadata to every chunk.

Every chunk carries the source filename plus the party names parsed from the
document text. Party metadata is load-bearing here: the corpus contains
near-identical indemnification amendments distinguished ONLY by party, so a
chunk that doesn't know its party is indistinguishable from its siblings.

Party extraction strategy (in order):
  1. Defined terms — legal drafting introduces parties as `Name (the "Role")`,
     e.g. `David Okafor (the "Indemnitee")`. All such pairs are collected.
     The primary party is chosen by role priority (Indemnitee > Executive >
     ... > Company), which picks the individual counterparty in personal
     agreements and the subject company in the merger agreement.
  2. Recital fallback — if no defined terms exist (the NDA), parse the
     `between X and Y` recital and take the named entities, with the last
     one as primary.
"""

import re

import config
from load import load_documents

# `Name (the "Role")` — the name is a run of capitalized words (allowing
# Inc., LLC, &, hyphens). The run may wrap across ONE line break (names wrap
# mid-recital, e.g. `Northgate\nInstruments Inc.`) but not a blank line or a
# comma, so headings and `WHEREAS,` can't glue onto the name.
_WORD = r"[A-Z][A-Za-z0-9.&'-]*"
_SEP = r"(?:[ \t]+|[ \t]*\n[ \t]*)"
_DEFINED_PARTY = re.compile(
    rf"((?:{_WORD}{_SEP})*{_WORD})\s*"
    rf"\(\s*(?:the\s+)?[\"“]([A-Za-z][A-Za-z ]{{1,30}})[\"”]\s*\)"
)

# `between X and Y` up to a clause boundary — used only when no defined terms
# are found.
_RECITAL = re.compile(
    r"\bbetween\s+(.+?)\s+and\s+(.+?)(?=\s+to\s|\s*[,.;\n])",
    re.IGNORECASE | re.DOTALL,
)

# Defined terms that name the contract rather than a party.
_NON_PARTY_ROLES = {"agreement", "plan", "amendment", "lease"}

# Roles most likely to identify the document's distinguishing party, best
# first. Individuals (indemnitee/executive) outrank organizations.
_ROLE_PRIORITY = ("indemnitee", "executive", "employee", "landlord", "company")


def _clean_name(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip(" ,;")


def extract_parties(text: str) -> tuple[str, list[str]]:
    """Return (primary_party, all_parties) parsed from a document's text."""
    by_role: dict[str, str] = {}
    for match in _DEFINED_PARTY.finditer(text):
        name, role = _clean_name(match.group(1)), match.group(2).strip().lower()
        if role in _NON_PARTY_ROLES:
            continue
        by_role.setdefault(role, name)

    parties = list(dict.fromkeys(by_role.values()))
    primary = ""
    for role in _ROLE_PRIORITY:
        if role in by_role:
            primary = by_role[role]
            break

    if not parties:
        recital = _RECITAL.search(text)
        if recital:
            parties = [_clean_name(g) for g in recital.groups()]
            primary = parties[-1]

    if not primary and parties:
        primary = parties[0]
    return primary, parties


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip("\n") for p in re.split(r"\n\s*\n", text) if p.strip()]


def _hard_split(paragraph: str) -> list[str]:
    """Window-split a paragraph that exceeds CHUNK_MAX_CHARS, with overlap."""
    step = config.CHUNK_MAX_CHARS - config.CHUNK_OVERLAP_CHARS
    return [
        paragraph[start : start + config.CHUNK_MAX_CHARS]
        for start in range(0, len(paragraph), step)
    ]


def chunk_text(text: str) -> list[str]:
    """Pack adjacent paragraphs up to CHUNK_TARGET_CHARS per chunk."""
    pieces = []
    for paragraph in _split_paragraphs(text):
        if len(paragraph) > config.CHUNK_MAX_CHARS:
            pieces.extend(_hard_split(paragraph))
        else:
            pieces.append(paragraph)

    chunks, current = [], ""
    for piece in pieces:
        if current and len(current) + len(piece) > config.CHUNK_TARGET_CHARS:
            chunks.append(current)
            current = piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def chunk_documents(docs: list[dict]) -> list[dict]:
    chunks = []
    for doc in docs:
        primary, parties = extract_parties(doc["text"])
        for i, text in enumerate(chunk_text(doc["text"])):
            chunks.append(
                {
                    "id": f"{doc['source']}#c{i}",
                    "source": doc["source"],
                    "chunk_index": i,
                    "party": primary,
                    "parties": parties,
                    "text": text,
                }
            )
    return chunks


if __name__ == "__main__":
    all_chunks = chunk_documents(load_documents())
    by_source: dict[str, list[dict]] = {}
    for c in all_chunks:
        by_source.setdefault(c["source"], []).append(c)
    print(f"{len(all_chunks)} chunks from {len(by_source)} documents\n")
    for source, doc_chunks in by_source.items():
        first = doc_chunks[0]
        print(f"{source}")
        print(f"  party:   {first['party'] or '(none)'}")
        print(f"  parties: {', '.join(first['parties']) or '(none)'}")
        print(f"  chunks:  {[len(c['text']) for c in doc_chunks]} chars")
