"""Premise lookup: map a drug pair to its gold DDI-2013 corpus sentence.

The NLI model is a sentence-pair classifier. For every atomic claim (hypothesis)
we need a premise = the ground-truth corpus sentence that mentions the same two
drugs. This module builds that lookup from the *_raw_manifest.csv files produced
by 0_ExtractData.ipynb, and gives the drug lists that constrain the UI
(the "Input Terbatas" requirement: only drugs with corpus coverage are offered).

Two filters keep the demo meaningful:
  - pairs with interaction_type == "none" are dropped: their sentence does not state
    an interaction between the two drugs (e.g. a brand name and its own generic);
  - Drug 1 only lists drugs with at least MIN_PAIRS interacting pairs, which removes
    rare spelling variants ("R(+) warfarin", "S(-)warfarin"). Drug 2 still lists every
    interacting partner of the chosen Drug 1.
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

MIN_PAIRS = 3  # interacting pairs a drug needs to appear in the Drug 1 list

# Manifests live at the repo root (one directory above app/), read in train/val/test order.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MANIFESTS = [os.path.join(_REPO_ROOT, f"{split}_raw_manifest.csv") for split in ("train", "val", "test")]

# Strip the [DRUGBANK] / [MEDLINE] source tag the same way the notebooks do.
_TAG_RE = re.compile(r"^\s*\[(?:DRUGBANK|MEDLINE)\]\s*", re.IGNORECASE)


def _norm_drug(name: str) -> str:
    """Lower-case, collapse whitespace: the notebooks' drug-name normalisation."""
    return " ".join(str(name).lower().split())


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    """Order-independent key so (A, B) and (B, A) find the same premise."""
    return tuple(sorted((_norm_drug(a), _norm_drug(b))))


@lru_cache(maxsize=1)
def _load():
    """Returns (pair -> premise, normalised name -> display name, normalised name -> partner set)."""
    frames = [pd.read_csv(p, usecols=["e1_text", "e2_text", "premise", "interaction_type"])
              for p in _MANIFESTS if os.path.exists(p)]
    if not frames:
        raise FileNotFoundError("No *_raw_manifest.csv found at the repo root. Run 0_ExtractData.ipynb first.")
    rows = pd.concat(frames, ignore_index=True).dropna(subset=["e1_text", "e2_text", "premise"])
    rows = rows[rows["interaction_type"].astype(str).str.lower() != "none"]

    pair_to_premise: Dict[Tuple[str, str], str] = {}
    display: Dict[str, str] = {}
    partners: Dict[str, Set[str]] = defaultdict(set)
    for e1, e2, premise in zip(rows["e1_text"], rows["e2_text"], rows["premise"]):
        e1, e2 = str(e1).strip(), str(e2).strip()
        a, b = _norm_drug(e1), _norm_drug(e2)
        if a == b:
            continue
        pair_to_premise.setdefault(_pair_key(e1, e2), _TAG_RE.sub("", str(premise)).strip())
        display.setdefault(a, e1)
        display.setdefault(b, e2)
        partners[a].add(b)
        partners[b].add(a)
    return pair_to_premise, display, partners


def drug_choices() -> List[str]:
    """Drug 1 list: drugs with at least MIN_PAIRS interacting pairs, sorted."""
    _, display, partners = _load()
    return sorted((display[d] for d, ps in partners.items() if len(ps) >= MIN_PAIRS), key=str.lower)


def partners_of(drug: str) -> List[str]:
    """Drug 2 list: every drug the corpus says interacts with `drug`."""
    _, display, partners = _load()
    return sorted((display[p] for p in partners.get(_norm_drug(drug), ())), key=str.lower)


def get_premise(drug1: str, drug2: str) -> Optional[str]:
    """Gold corpus sentence stating the interaction of this pair, or None if there is none."""
    return _load()[0].get(_pair_key(drug1, drug2))
