"""Inference pipeline: atomic claims -> NLI prediction -> aggregate factuality.

Chains the three deployment stages (the Gemini call is added separately in app.py):
  1. atomiser.atomize_text   -> atomic SVO claims (hypotheses)
  2. this module's predict   -> entailment / contradiction / neutral per claim
  3. aggregate               -> one verdict + a FActScore-style factuality score

The NLI model is the saved atomic_pubmedbert/ (BertForSequenceClassification).
Tokenisation reproduces 4a/4b exactly: tokenizer(premise, hypothesis,
truncation='only_second', max_length=512), with the premise pre-capped so the
hypothesis always keeps at least 32 tokens of budget.
"""
from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from . import atomiser
from .premise_lookup import get_premise

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_DIR = os.path.join(_REPO_ROOT, "atomic_pubmedbert")
_MIN_HYPOTHESIS_BUDGET = 32  # 4a's BERTNLIDataset.MIN_HYPOTHESIS_BUDGET
_MAX_LEN = 512

_tokenizer = None
_model = None
_id2label: Dict[int, str] = {}


@dataclass
class AtomResult:
    hypothesis: str          # the atomic claim text
    subject: str
    relation: str
    object: str
    label: str               # entailment / contradiction / neutral
    probs: Dict[str, float]  # softmax confidence per class


@dataclass
class VerificationResult:
    premise: str
    answer: str
    atoms: List[AtomResult] = field(default_factory=list)
    verdict: str = "neutral"          # aggregate answer-level label
    factscore: float = 0.0            # fraction of atoms the premise supports (entailment)
    label_counts: Dict[str, int] = field(default_factory=dict)
    note: str = ""                    # e.g. "no premise for this pair" / "no atoms extracted"


def _load_model():
    global _tokenizer, _model, _id2label
    if _model is not None:
        return
    _tokenizer = AutoTokenizer.from_pretrained(_MODEL_DIR)
    _model = AutoModelForSequenceClassification.from_pretrained(_MODEL_DIR).eval()
    # config.json: {0: entailment, 1: contradiction, 2: neutral}
    _id2label = {int(k): v for k, v in _model.config.id2label.items()}


def _cap_premise(premise: str) -> str:
    """Pre-trim the premise to max_len - 32 tokens (4a's _capped_premise)."""
    max_premise_len = max(_MAX_LEN - _MIN_HYPOTHESIS_BUDGET, 1)
    ids = _tokenizer.encode(premise, add_special_tokens=False)
    if len(ids) <= max_premise_len:
        return premise
    return _tokenizer.decode(ids[:max_premise_len])


@torch.no_grad()
def predict_pairs(premise: str, hypotheses: List[str]) -> List[Dict]:
    """Classify each (premise, hypothesis) pair. Returns [{label, probs}, ...]."""
    _load_model()
    capped = _cap_premise(premise)
    results = []
    for hyp in hypotheses:
        enc = _tokenizer(capped, hyp, max_length=_MAX_LEN, truncation="only_second", return_tensors="pt")
        logits = _model(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1)
        idx = int(probs.argmax())
        results.append({
            "label": _id2label[idx],
            "probs": {_id2label[i]: float(probs[i]) for i in range(len(probs))},
        })
    return results


def aggregate(labels: List[str]) -> str:
    """Strict rule (AtomicAggregator, AGG_RULE='strict'): any contradiction -> C,
    else any neutral -> N, else entailment."""
    if "contradiction" in labels:
        return "contradiction"
    if "neutral" in labels:
        return "neutral"
    return "entailment"


def verify(answer: str, drug1: str, drug2: str, premise: Optional[str] = None) -> VerificationResult:
    """Full verification of a free-text answer about (drug1, drug2).

    premise: pass an explicit premise, or leave None to look up the gold DDI-2013
    sentence for the drug pair. Without a premise no NLI is possible."""
    if premise is None:
        premise = get_premise(drug1, drug2)

    result = VerificationResult(premise=premise or "", answer=answer)

    if not premise:
        result.note = f"No DDI-2013 premise for the pair ({drug1}, {drug2}); cannot verify."
        return result

    atoms_df = atomiser.atomize_text(answer, drug1, drug2)
    if atoms_df.empty:
        result.note = "No atomic claims could be extracted from the answer."
        return result

    hypotheses = atoms_df["hypothesis"].tolist()
    preds = predict_pairs(premise, hypotheses)

    for (_, row), pred in zip(atoms_df.iterrows(), preds):
        result.atoms.append(AtomResult(
            hypothesis=row["hypothesis"],
            subject=row["sub_extract"], relation=row["rel_extract"], object=row["obj_extract"],
            label=pred["label"], probs=pred["probs"],
        ))

    labels = [a.label for a in result.atoms]
    result.label_counts = dict(Counter(labels))
    result.verdict = aggregate(labels)
    result.factscore = labels.count("entailment") / len(labels)
    return result
