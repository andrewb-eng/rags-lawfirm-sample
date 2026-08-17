"""Stage 2 — chunk: split documents into section-aware chunks with metadata.

Every chunk carries the source filename, the party names parsed from the
document text, and the legal section heading it falls under. Party metadata is
load-bearing: the needle corpus contains near-identical indemnification
amendments distinguished ONLY by party, so a chunk that doesn't know its party
is indistinguishable from its siblings.

Chunking pipeline (built for both tidy synthetic docs and PDF-extracted CUAD
contracts, whose newlines are collapsed into runs of spaces):

  1. normalize   — drop standalone page-number lines
  2. blocks      — split on blank lines
  3. pieces      — split blocks again where a section marker appears inline
                   after a run of 3+ spaces (the PDF-extraction signature),
                   e.g. "...agree as follows:     1. DEFINITIONS. ..."
  4. pack        — greedily merge pieces up to CHUNK_TARGET_CHARS, but never
                   across a section boundary; pieces longer than
                   CHUNK_MAX_CHARS are split at sentence ends

Party extraction strategy (in order):
  1. Defined terms — `Name (the "Role")`, role-prioritized so the individual
     counterparty in personal agreements outranks the company.
  2. Quoted short-name aliases — real contracts define parties after an
     address blob (`..., a Delaware corporation ("Accuray")`), where no
     capitalized name precedes the paren. The quoted name itself is taken,
     filtered against a stoplist of boilerplate defined terms and generic
     role words.
  3. Recital fallback — `by and between/among X and Y` when nothing else
     matched; names are cut at the first comma so address blobs don't leak
     in, and the last party is primary.
"""

import re

import config
from load import load_documents

# --- party extraction -------------------------------------------------------

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

# `..., a Delaware corporation ("Accuray")` — capture the quoted short name
# regardless of what precedes the paren.
_QUOTED_ALIAS = re.compile(
    r"\(\s*(?:the\s+)?[\"“]([A-Z][A-Za-z0-9 .&'-]{1,40})[\"”]\s*\)"
)

# `by and between X and Y` (or `among`), searched near the top of the
# document; a bare `between X and Y` is the last resort. The first two
# patterns require a lowercase `and` as the party connector so that ALL-CAPS
# names containing "AND" (INSTITUTE OF GRASSLAND AND ...) don't get cut; the
# case-insensitive forms are the fallback for fully-capitalized recitals.
_RECITAL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (
        r"\bby\s+and\s+(?:between|among)\s+(.+?)\s+(?-i:and)\s+(.+?)(?=\s+to\s|\s*[,.;\n])",
        r"\bbetween\s+(.+?)\s+(?-i:and)\s+(.+?)(?=\s+to\s|\s*[,.;\n])",
        r"\bby\s+and\s+(?:between|among)\s+(.+?)\s+and\s+(.+?)(?=\s+to\s|\s*[,.;\n])",
        r"\bbetween\s+(.+?)\s+and\s+(.+?)(?=\s+to\s|\s*[,.;\n])",
    )
]

# A defined term counts as a party only when its quoted role is a word
# contracts actually use for parties. Anything else ("Territory", "Merger",
# "Effective Time"...) is a boilerplate definition, not a party.
_PARTY_ROLES = {
    "indemnitee", "executive", "employee", "landlord", "tenant", "company",
    "parent", "subsidiary", "distributor", "customer", "supplier", "buyer",
    "seller", "licensee", "licensor", "purchaser", "vendor", "client",
    "borrower", "lender", "contractor", "consultant", "guarantor", "agent",
    "investor", "member", "partner", "owner", "operator", "manufacturer",
    "developer", "provider", "recipient", "sponsor", "reseller",
    "franchisee", "franchisor", "lessee", "lessor", "holder", "issuer",
    "trustee", "stockholder", "shareholder", "advisor", "adviser",
}

# Words that end document names; a "party" name ending in one is boilerplate
# (e.g. `this Distribution Agreement (the "Company")`).
_DOC_NAME_TAILS = {
    "agreement", "amendment", "plan", "lease", "contract", "addendum",
    "annex", "exhibit", "schedule", "note", "guaranty",
}

# Roles most likely to identify the document's distinguishing party, best
# first. Individuals (indemnitee/executive) outrank organizations.
_ROLE_PRIORITY = ("indemnitee", "executive", "employee", "landlord", "company")

# Quoted aliases that are boilerplate defined terms or generic role words,
# not party names. Lowercased for comparison.
_ALIAS_STOP = {
    "agreement", "amendment", "amendments", "plan", "lease", "sublease",
    "guaranty", "note", "merger", "merger sub", "merger agreement",
    "effective date", "effective time", "agreement date", "execution date",
    "closing date", "closing", "term", "renewal term", "initial term",
    "territory", "products", "product", "services", "service", "software",
    "system", "systems", "specifications", "confidential information",
    "purchase price", "warranty", "warranty period", "section", "exhibit",
    "schedule", "appendix", "annex", "party", "parties", "business",
    "company", "distributor", "customer", "supplier", "buyer", "seller",
    "licensee", "licensor", "landlord", "tenant", "executive", "employee",
    "indemnitee", "contractor", "consultant", "purchaser", "vendor",
    "client", "borrower", "lender", "parent", "subsidiary", "guarantor",
    "agent", "trustee", "investor", "stockholder", "stockholders",
    "shareholder", "member", "partner", "owner", "operator", "manufacturer",
    "developer", "provider", "recipient", "sponsor", "reseller",
    "franchisee", "franchisor", "assignee", "assignor", "holder", "issuer",
    "lessee", "lessor", "bank", "escrow agent", "representative",
}


def _clean_name(name: str) -> str:
    name = re.sub(r"\s+", " ", name).strip(" ,;")
    return re.sub(r"^(?:the|The)\s+", "", name)


def _recital_parties(text: str) -> list[str]:
    """Parse party names from the opening recital, cutting address blobs."""
    head = text[:3000]
    for pattern in _RECITAL_PATTERNS:
        match = pattern.search(head)
        if not match:
            continue
        names = []
        for group in match.groups():
            name = _clean_name(group.split(",")[0])
            if name and name[0].isupper() and len(name) <= 60:
                names.append(name)
        if len(names) == 2:
            return names
    return []


def extract_parties(text: str) -> tuple[str, list[str]]:
    """Return (primary_party, all_parties) parsed from a document's text."""
    by_role: dict[str, str] = {}
    for match in _DEFINED_PARTY.finditer(text):
        name, role = _clean_name(match.group(1)), match.group(2).strip().lower()
        if role not in _PARTY_ROLES:
            continue
        if name.lower() in _ALIAS_STOP or name.split()[-1].lower() in _DOC_NAME_TAILS:
            continue
        by_role.setdefault(role, name)

    parties = list(dict.fromkeys(by_role.values()))
    primary = ""
    for role in _ROLE_PRIORITY:
        if role in by_role:
            primary = by_role[role]
            break

    known = {p.lower() for p in parties}
    for match in _QUOTED_ALIAS.finditer(text):
        alias = _clean_name(match.group(1))
        low = alias.lower()
        if low in _ALIAS_STOP or low in known or len(alias.split()) > 5:
            continue
        parties.append(alias)
        known.add(low)

    if not parties:
        parties = _recital_parties(text)
        if parties:
            primary = parties[-1]

    if not primary and parties:
        primary = parties[0]
    return primary, parties


# --- chunking ----------------------------------------------------------------

_PAGE_LINE = re.compile(r"^\s*(?:page\s+)?-?\s*\d{1,4}\s*-?\s*$", re.IGNORECASE)
_CAPS_HEADING = re.compile(r"^\s*[A-Z][A-Z0-9 .,:&()'\"/–—-]{3,80}\s*$")

# Section markers: "ARTICLE IV", "SECTION 3", "1. DEFINITIONS.",
# "2.1. Appointment.", '1.1. "Region" means...'. The number form requires the
# trailing . or ) so years and quantities don't match.
_MARKER = (
    r"(?:ARTICLE|SECTION|EXHIBIT|SCHEDULE|APPENDIX|ANNEX)\s+[IVXLC0-9][A-Za-z0-9.]*"
    r"|(?:Section\s+)?\d{1,2}(?:\.\d{1,2}){0,3}[.)]\s+[\"“(A-Z]"
)
_SECTION_START = re.compile(rf"^\s*(?:{_MARKER})")
# Inline markers only count after a run of 3+ spaces — the signature of a
# collapsed newline in PDF-extracted text. "pursuant to Section 2.3 of ..."
# (single space) never splits.
_INLINE_SPLIT = re.compile(rf"\s{{3,}}(?=(?:{_MARKER}))")
_SECTION_LABEL = re.compile(
    r"^\s*((?:ARTICLE|SECTION|EXHIBIT|SCHEDULE|APPENDIX|ANNEX)\s+[IVXLC0-9][A-Za-z0-9.]*"
    r"|(?:Section\s+)?\d{1,2}(?:\.\d{1,2}){0,3}[.)]\s+[^.\n]{0,60})"
)
_SENTENCE_END = re.compile(r"(?<=[.;:])\s+")


def _normalize(text: str) -> str:
    lines = [ln for ln in text.splitlines() if not _PAGE_LINE.match(ln)]
    return "\n".join(lines)


def _section_label(piece: str) -> str:
    """Heading label for a piece that starts a section, or ''."""
    first_line = piece.lstrip().splitlines()[0] if piece.strip() else ""
    if _CAPS_HEADING.match(first_line) and len(first_line.strip()) <= 80:
        return re.sub(r"\s+", " ", first_line).strip()
    match = _SECTION_LABEL.match(piece)
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip(" .")
    return ""


def _split_long(piece: str) -> list[str]:
    """Split an over-long piece at sentence ends near CHUNK_TARGET_CHARS."""
    sentences = _SENTENCE_END.split(piece)
    out, current = [], ""
    for sentence in sentences:
        while len(sentence) > config.CHUNK_MAX_CHARS:  # pathological: no breaks
            if current:
                out.append(current)
                current = ""
            step = config.CHUNK_MAX_CHARS - config.CHUNK_OVERLAP_CHARS
            out.append(sentence[: config.CHUNK_MAX_CHARS])
            sentence = sentence[step:]
        if current and len(current) + len(sentence) + 1 > config.CHUNK_TARGET_CHARS:
            out.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence
    if current:
        out.append(current)
    return out


def _pieces(text: str) -> list[tuple[str, bool]]:
    """(piece, starts_section) tuples, in document order."""
    pieces = []
    for block in re.split(r"\n\s*\n", _normalize(text)):
        if not block.strip():
            continue
        for i, part in enumerate(_INLINE_SPLIT.split(block)):
            part = part.strip("\n")
            if not part.strip():
                continue
            starts = bool(
                _SECTION_START.match(part)
                or _CAPS_HEADING.match(part.lstrip().splitlines()[0])
            )
            # collapse intra-piece space runs left over from PDF extraction
            part = re.sub(r"[ \t]{2,}", " ", part)
            if len(part) > config.CHUNK_MAX_CHARS:
                for j, sub in enumerate(_split_long(part)):
                    pieces.append((sub, starts and j == 0))
            else:
                pieces.append((part, starts))
    return pieces


def chunk_text(text: str) -> list[dict]:
    """Pack pieces into chunks of ~CHUNK_TARGET_CHARS within one section."""
    chunks: list[dict] = []
    current, section = "", ""

    def flush():
        nonlocal current
        if current.strip():
            chunks.append({"text": current, "section": section})
        current = ""

    for piece, starts_section in _pieces(text):
        if starts_section:
            flush()
            section = _section_label(piece) or section
        elif current and len(current) + len(piece) > config.CHUNK_TARGET_CHARS:
            flush()
        current = f"{current}\n\n{piece}" if current else piece
    flush()
    return chunks


def chunk_documents(docs: list[dict]) -> list[dict]:
    chunks = []
    for doc in docs:
        primary, parties = extract_parties(doc["text"])
        for i, piece in enumerate(chunk_text(doc["text"])):
            chunks.append(
                {
                    "id": f"{doc['source']}#c{i}",
                    "source": doc["source"],
                    "chunk_index": i,
                    "party": primary,
                    "parties": parties,
                    "section": piece["section"],
                    "text": piece["text"],
                }
            )
    return chunks


if __name__ == "__main__":
    all_chunks = chunk_documents(load_documents())
    by_source: dict[str, list[dict]] = {}
    for c in all_chunks:
        by_source.setdefault(c["source"], []).append(c)
    print(f"{len(all_chunks)} chunks from {len(by_source)} documents\n")
    for source, doc_chunks in list(by_source.items())[:20]:
        first = doc_chunks[0]
        print(f"{source}")
        print(f"  party:   {first['party'] or '(none)'}")
        print(f"  parties: {', '.join(first['parties']) or '(none)'}")
        print(f"  chunks:  {[len(c['text']) for c in doc_chunks]} chars")
