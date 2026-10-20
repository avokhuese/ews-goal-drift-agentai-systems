"""Structured source that the agent queries against."""

import json
from pathlib import Path
from typing import List, Dict, Any


DEFAULT_DOCS = [
    {"id": "d1", "topic": "sea_level", "text": "Thermal expansion of ocean water accounts for roughly 40% of observed sea-level rise since 1970."},
    {"id": "d2", "topic": "sea_level", "text": "Glacier melt in the Himalayas and Andes contributes a measurable fraction of sea-level rise."},
    {"id": "d3", "topic": "sea_level", "text": "Greenland ice sheet mass loss has accelerated since the 1990s, contributing ~0.7 mm/yr."},
    {"id": "d4", "topic": "sea_level", "text": "Antarctic ice sheet dynamics remain the largest uncertainty in long-term projections."},
    {"id": "d5", "topic": "sea_level", "text": "Satellite altimetry since 1993 shows a global mean sea-level rise of 3.4 mm/yr."},
    {"id": "d6", "topic": "sea_level", "text": "Ocean heat content is a dominant driver of steric sea-level change."},
    {"id": "d7", "topic": "sea_level", "text": "Tide gauge records extend the observational baseline back to the 19th century."},
    {"id": "d8", "topic": "sea_level", "text": "Groundwater depletion adds roughly 0.4 mm/yr to sea-level rise."},
    {"id": "e1", "topic": "economics", "text": "Coastal real estate markets are repricing risk in Florida and California."},
    {"id": "e2", "topic": "economics", "text": "Flood insurance premiums in the US have risen sharply since 2020."},
    {"id": "e3", "topic": "economics", "text": "Municipal bond yields in coastal cities reflect climate risk perceptions."},
    {"id": "e4", "topic": "economics", "text": "Property tax base erosion is a growing concern for coastal municipalities."},
    {"id": "n1", "topic": "noise", "text": "The mating habits of Antarctic krill are poorly understood."},
    {"id": "n2", "topic": "noise", "text": "The history of lighthouse construction is a niche academic field."},
]


def build_default_corpus(path: str = "data/corpus.jsonl", force: bool = False) -> None:
    """Create a small structured corpus if it doesn't exist OR is empty."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Rebuild if the file is missing, empty, or we are forcing a refresh.
    needs_build = force or (not p.exists()) or (p.stat().st_size == 0)
    if not needs_build:
        # Sanity check: file has content but may be malformed → treat as needs_build
        try:
            with p.open("r") as f:
                first = f.readline().strip()
            if not first:
                needs_build = True
        except Exception:
            needs_build = True

    if not needs_build:
        return

    with p.open("w") as f:
        for d in DEFAULT_DOCS:
            f.write(json.dumps(d) + "\n")
    print(f"[corpus] Wrote {len(DEFAULT_DOCS)} documents to {p}")


def load_corpus(path: str = "data/corpus.jsonl") -> List[Dict[str, Any]]:
    build_default_corpus(path)
    docs = []  # type: List[Dict[str, Any]]
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    if not docs:
        raise RuntimeError(
            f"Corpus at {path} is empty after build. "
            f"Delete the file and rerun, or check write permissions."
        )
    return docs