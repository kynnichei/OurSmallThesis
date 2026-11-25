"""Atomic-claim decomposition (SVO triplets) — lifted verbatim from 2_Atomisation_patched.ipynb.

This is the Rule-Based Dependency Parser stage: a scispaCy (en_core_sci_sm) pipeline
breaks a free-text medical answer into atomic claims, one per drug pair / coordinated
predicate, each with its subject, relation and object. The public entry point is
`atomize_text(text, drug1, drug2)` which wraps the notebook's `atomize_dataframe`.

The drug vocabulary and PhraseMatcher are built lazily from *_raw_manifest.csv so the
module imports cheaply and the heavy spaCy load only happens on first use.
"""
from __future__ import annotations

import ast
import json
import os
import re
from collections import Counter, defaultdict

import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
from spacy.util import filter_spans

# ── configuration (identical to the notebook) ──────────────────────────────────────────────────
NEUTRAL_SAME_PAIR = True
SPLIT_CONJUNCTS = True
KEEP_CONTEXT_PHRASES = True
MIN_LEN, MAX_LEN = 4, 30
FALLBACK_MAX_LEN = 40
LABELS = ("entailment", "contradiction", "neutral")
ALLOW_FALLBACK = False  # set True only for a smoke test without scispaCy installed

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MANIFESTS = [os.path.join(_REPO_ROOT, f"{s}_raw_manifest.csv") for s in ("train", "val", "test")]

# ── lexicons (identical to the notebook) ────────────────────────────────────────────────────────
EFFECT_BASE = {
 "increase","decrease","reduce","enhance","potentiate","inhibit","raise","lower","prolong","elevate","diminish",
 "attenuate","augment","boost","amplify","suppress","accelerate","slow","impair","induce","block","antagonize",
 "intensify","strengthen","weaken","limit","alter","modify","affect","cause","exacerbate","worsen","improve",
 "extend","shorten","abolish","reverse","counteract","mask","lengthen","heighten","displace","metabolize",
 "interfere","interact","compete","bind","stimulate","precipitate","trigger","promote","mediate","delay",
 "hasten","oppose","offset","neutralize","produce","result","lead","change","clear","excrete","absorb",
 "reduction","inhibition","elevation","potentiation","enhancement","induction","interaction","rise","decline",
 "displacement","competition","interference","antagonism","synergism","sensitize","predispose","contribute",
 "minimize","maximize","depress","dampen","hinder","facilitate","influence","speed","prevent","disrupt",
}
ADVISE_BASE = {"avoid","contraindicate","contraindication","contraindicated","advisable","recommend","prohibit",
               "discourage","warn","caution","monitor","adjust","require","necessitate","dangerous","unsafe"}
NO_CHANGE_BASE = {"unchanged","unaffected","unaltered","toxicity","link","associate","additive","synergistic",
                  "disruption","drop","fail","lack","ineffective","opposed"}
EFFECT_ADDED = {"cut", "halve"}
ADVISE_ADDED = {"mandate", "advise", "encourage", "support", "endorse", "urge", "instruct", "need"}
NO_CHANGE_ADDED = {"difference", "same", "maintain", "remain", "stay", "constant", "equivalent", "comparable",
                   "finding", "target"}
EFFECT = EFFECT_BASE | EFFECT_ADDED
ADVISE = ADVISE_BASE | ADVISE_ADDED
NO_CHANGE_ETC = NO_CHANGE_BASE | NO_CHANGE_ADDED
STRONG = EFFECT | ADVISE | NO_CHANGE_ETC
LIGHT = {"lead", "result", "contribute", "cause", "produce", "predispose", "link", "associate"}
COADMIN_VERBS = {"combine", "coadminister", "co-administer", "administer", "take", "give", "use", "prescribe", "mix"}
ADVICE_MODALS = {"should", "must", "shall", "ought"}

ORDINARY_MIN_SEEN, ORDINARY_MAX_RATE = 3, 0.5
DESCRIPTOR_NOUNS = {"effect", "effects", "action", "actions", "activity", "activities", "property", "properties",
                    "potency", "efficacy", "response", "responses"}
GENERIC_STOP = {
    "lead","sodium","acid","agent","agents","drug","drugs","combination","type","d","n","vitamin","iron",
    "calcium","thyroid","peripheral","urinary","antibiotic","antibiotics","therapy","treatment","levels",
    "inhibitor","inhibitors","blocker","blockers","antagonist","antagonists","hydrochloride","sulfate",
    "oral","water","salt","alcohol","caffeine","folic","ethinyl","gel","cream","tablet","tablets",
    "potassium","magnesium","glucose","zinc","oxygen","food","foods","phosphate","chloride",
}

CONTEXT_PREPS   = {"during","for","in","after","before","throughout","within","among","at","under","upon","since","over"}
CONDITION_MARKS = {"if","unless","when","whenever","while","once","until","provided"}
KEEP_DEPS = {"nsubj","nsubjpass","csubj","csubjpass","dobj","attr","acomp","xcomp","ccomp","agent","dative","oprd",
             "neg","aux","auxpass","prt","advmod","npadvmod","expl","mark","prep","advcl","cc","conj","relcl","acl","appos"}
ACTOR_DEPS      = {"nsubj","csubj","agent"}
SUBJ_DEPS       = {"nsubj","nsubjpass","csubj","csubjpass","expl"}
SHARED_AUX_DEPS = {"aux","auxpass","neg"}
COMPLEMENT_DEPS = {"prep","dobj","attr","acomp","xcomp","ccomp","oprd","dative","agent","dep","pobj"}
STOP_DEPS       = {"advcl","conj","cc","relcl","parataxis","mark","punct"}
CONTENT_POS     = ("VERB","NOUN","ADJ","AUX","ADV")
LEMMA_FIX = {"coadministere": "coadminister", "co-administere": "co-administer", "administere": "administer"}
CLASS_NOUNS = ("drugs", "drug", "agents", "agent", "medications", "medication", "medicines", "medicine")

ATOM_FIELDS = ["sub_extract", "rel_extract", "rel_lemma", "obj_extract", "role_source", "hypothesis",
               "source_tier", "repaired", "context_trimmed", "trigger_words", "other_drugs"]
OUT_COLUMNS = ["source_row", "original_id", "item_id", "gen_mode", "Entity1", "Entity2", "premise", "source_text",
               "response_label", "claim_idx", "claim_text", "label", "label_source"] + ATOM_FIELDS + ["filter_reason"]

# ── lazily-built global state (spaCy pipeline + drug vocabulary + matcher) ───────────────────────
nlp = None
VOCAB: set = set()
ORDINARY: set = set()
global_matcher: PhraseMatcher = None
_READY = False


def _norm(text):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split())


def _ordinary_names(rows, names):
    index = defaultdict(list)
    for n in {_norm(n) for n in names}:
        if n:
            index[n.split()[0]].append(tuple(n.split()))
    for cands in index.values():
        cands.sort(key=len, reverse=True)
    own = defaultdict(set)
    for a, b, premise in zip(rows["e1_text"], rows["e2_text"], rows["premise"]):
        own[re.sub(r"^\[\w+\]\s*", "", premise).strip()].update((_norm(a), _norm(b)))
    seen, marked = Counter(), Counter()
    for body, drugs in own.items():
        words, i = _norm(body).split(), 0
        while i < len(words):
            for cand in index.get(words[i], ()):
                if tuple(words[i:i + len(cand)]) == cand:
                    name = " ".join(cand)
                    if i + len(cand) >= len(words) or words[i + len(cand)] not in DESCRIPTOR_NOUNS:
                        seen[name] += 1
                        marked[name] += name in drugs
                    i += len(cand) - 1
                    break
            i += 1
    return {n for n, k in seen.items() if k >= ORDINARY_MIN_SEEN and marked[n] / k < ORDINARY_MAX_RATE}


def _load_parser():
    try:
        return spacy.load("en_core_sci_sm")
    except OSError:
        if not ALLOW_FALLBACK:
            raise RuntimeError(
                "en_core_sci_sm is not installed. Install scispaCy and en_core_sci_sm "
                "(or set atomiser.ALLOW_FALLBACK = True to knowingly use en_core_web_sm)."
            )
        return spacy.load("en_core_web_sm")


def _ensure_ready():
    """Build the spaCy pipeline, drug vocabulary and PhraseMatcher once (lazy, ~first call)."""
    global nlp, VOCAB, ORDINARY, global_matcher, _READY
    if _READY:
        return
    nlp = _load_parser()

    found = [p for p in _MANIFESTS if os.path.exists(p)]
    if not found:
        raise FileNotFoundError(f"None of {_MANIFESTS} found: run 0_ExtractData.ipynb first.")
    rows = pd.concat([pd.read_csv(p, usecols=["e1_text", "e2_text", "premise"]) for p in found]).dropna().astype(str)
    drug_names = {" ".join(n.lower().split()) for n in pd.concat([rows["e1_text"], rows["e2_text"]])}
    drug_names = {n for n in drug_names if len(n) > 2 and not n.isdigit()}

    ORDINARY = _ordinary_names(rows, drug_names)

    vocab = set()
    for _n in drug_names:
        vocab.add(_n)
        vocab.update(w for w in _n.split() if len(w) > 2)
    vocab -= GENERIC_STOP

    def _forms(w):
        return {w, w[:-1], w[:-2], w[:-3], w[:-3] + "e", w[:-2] + "e"} if len(w) > 4 else {w}
    VOCAB = {w for w in vocab if w in drug_names or not (_forms(w) & STRONG)}

    patterns = [nlp.make_doc(t) for t in drug_names if t not in GENERIC_STOP and _norm(t) not in ORDINARY]
    global_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    global_matcher.add("CORPUS_DRUGS", patterns)
    _READY = True


# ── text helpers ─────────────────────────────────────────────────────────────────────────────
def lemma(tok):
    l = tok.lemma_.lower()
    return LEMMA_FIX.get(l, l)


def sentence_case(s):
    s = re.sub(r"\s+([,.;:)])", r"\1", s.strip())
    s = re.sub(r"\(\s+", "(", s)
    return s[:1].upper() + s[1:] + ("" if s.endswith(".") else ".")


def surface(toks):
    ids = {t.i for t in toks}
    doc = toks[0].doc
    for t in toks[:-1]:
        nxt = t.i + 1
        if (nxt + 1 in ids and nxt not in ids and doc[nxt].text in ("-", "/")
                and not t.whitespace_ and not doc[nxt].whitespace_):
            ids.add(nxt)
    out, prev = [], None
    for i in sorted(ids):
        if prev is not None:
            out.append(doc[prev].whitespace_ if i == prev + 1 else " ")
        out.append(doc[i].text)
        prev = i
    return sentence_case("".join(out))


# ── drug mentions ─────────────────────────────────────────────────────────────────────────────
def name_variants(name):
    name = name.lower().strip()
    variants = {name}
    words = name.split()
    if not words:
        return variants
    head, last = words[:-1], words[-1]
    if len(last) >= 5:
        variants.add(" ".join(head + [last[:-1] if last.endswith("s") and not last.endswith("ss") else last + "s"]))
    if last in CLASS_NOUNS and head:
        variants.update(" ".join(head + [c]) for c in CLASS_NOUNS)
    return {v for v in variants if len(v) > 2}


def _labelled(name, known_pair):
    for k in known_pair:
        vs = name_variants(k)
        if name in vs:
            return True
        if any(len(v) >= 4 and v in name for v in vs):
            return True
        if len(name) >= 5 and name not in GENERIC_STOP and any(name in v for v in vs):
            return True
    return False


def canon(name, known_pair):
    ordered = sorted(known_pair, key=len, reverse=True)
    for k in ordered:
        if name in name_variants(k):
            return k
    for k in ordered:
        if _labelled(name, {k}):
            return k
    return name


def entity_spans(sent_doc, known_pair):
    pair_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    pair_matcher.add("KNOWN_PAIR", [nlp.make_doc(v) for n in known_pair for v in name_variants(n)])
    spans = [sent_doc[s:e] for _, s, e in list(global_matcher(sent_doc)) + list(pair_matcher(sent_doc))]
    first, mentions = [], defaultdict(list)
    for sp in filter_spans(spans):
        name = sp.text.lower()
        labelled = _labelled(name, known_pair)
        if not labelled:
            if name in GENERIC_STOP or len(name) <= 2 or sp.root.pos_ not in {"NOUN", "PROPN", "X"}:
                continue
            if sp.end < len(sp.doc) and sp.doc[sp.end].lower_ in DESCRIPTOR_NOUNS:
                continue
        key = canon(name, known_pair) if labelled else name
        if key not in mentions:
            first.append((sp, key))
        mentions[key].append(sp)
    return first, mentions


# ── predicate and clause ──────────────────────────────────────────────────────────────────────
def modal_advice(t):
    return lemma(t) in COADMIN_VERBS and any(c.dep_ == "aux" and c.text.lower() in ADVICE_MODALS for c in t.children)


def is_trigger(t):
    return t.pos_ in CONTENT_POS and (lemma(t) in STRONG or modal_advice(t))


def trigger_name(t):
    return lemma(t) if lemma(t) in STRONG else lemma(t) + "+modal"


def supports(p):
    return any(is_trigger(t) and t.dep_ != "advcl" for t in p.subtree)


def find_predicate(a, b):
    cur, seen = a.root, set()
    while True:
        if cur.head.i == cur.i:
            return None
        cur = cur.head
        if cur.i in seen:
            return None
        seen.add(cur.i)
        if b.root.i in {t.i for t in cur.subtree} and cur.pos_ in ("VERB", "AUX", "ADJ") and supports(cur):
            return cur


def is_condition(advcl):
    return any(ch.dep_ in ("mark", "advmod") and ch.text.lower() in CONDITION_MARKS for ch in advcl.children)


def core_clause(p, needed_idx, keep_context=False):
    light = lemma(p) in LIGHT
    drop, conj_kept = set(), False
    for c in p.children:
        has_need = any(t.i in needed_idx for t in c.subtree)
        if c.dep_ == "punct":
            drop.update(t.i for t in c.subtree)
        elif has_need:
            conj_kept |= c.dep_ == "conj"
            continue
        elif c.dep_ == "advcl":
            if not (keep_context and is_condition(c)):
                drop.update(t.i for t in c.subtree)
        elif c.dep_ == "prep" and c.text.lower() in CONTEXT_PREPS:
            if not (keep_context or light):
                drop.update(t.i for t in c.subtree)
        elif c.dep_ in ("conj", "cc", "parataxis", "relcl"):
            drop.update(t.i for t in c.subtree)
        elif c.dep_ not in KEEP_DEPS:
            drop.update(t.i for t in c.subtree)
    if conj_kept:
        drop -= {c.i for c in p.children if c.dep_ in ("cc", "preconj")}
    return [t for t in p.subtree if t.i not in drop and t.pos_ != "PUNCT"]


def conjunct_clause(p, c, needed_idx, keep_context=False, full=False):
    doc = c.doc
    if full:
        chain = {t.i for ch in c.children if ch.dep_ in ("conj", "cc", "preconj", "punct") for t in ch.subtree}
        ids = {t.i for t in c.subtree} - chain
    else:
        ids = {t.i for t in core_clause(c, needed_idx, keep_context)}
    if not any(ch.dep_ in SUBJ_DEPS for ch in c.children):
        for ch in p.children:
            if ch.dep_ in SUBJ_DEPS:
                ids |= {t.i for t in ch.subtree}
        if not is_finite(c):
            ids |= {ch.i for ch in p.children if ch.dep_ in SHARED_AUX_DEPS and ch.i < p.i}
    return [doc[i] for i in sorted(ids) if doc[i].pos_ != "PUNCT"]


def is_fragment(p, toks):
    last = toks[-1]
    if last.pos_ not in ("VERB", "AUX", "ADJ", "ADV"):
        return False
    kept = {t.i for t in toks}
    rest = [t for t in p.subtree if t.i > last.i and t.i not in kept and t.pos_ != "PUNCT"]
    if not rest:
        return False
    t = rest[0]
    return not (t.text.lower() in CONTEXT_PREPS or t.pos_ in ("SCONJ", "CCONJ")
                or t.dep_ in STOP_DEPS or t.tag_.startswith("W"))


def is_finite(p):
    FIN = {"VBZ", "VBP", "VBD", "MD"}
    if p.tag_ in FIN or any(c.dep_ in ("aux", "auxpass") and c.tag_ in FIN for c in p.children):
        return True
    return (p.tag_ == "VB" and p.dep_ in ("ROOT", "conj")
            and any(c.dep_ in SUBJ_DEPS for c in p.children)
            and not any(c.dep_ in ("aux", "auxpass") for c in p.children))


def triggers(toks, drug_idx):
    return [t for t in toks if is_trigger(t) and t.i not in drug_idx and t.text.lower() not in VOCAB]


def missing_complement(toks, drug_idx):
    trig = triggers(toks, drug_idx)
    if not trig or any(lemma(t) not in LIGHT for t in trig):
        return False
    ids = {t.i for t in toks}
    return not any(ch.i in ids and ch.dep_ in COMPLEMENT_DEPS for t in trig for ch in t.children)


def dangling(toks):
    return (toks[0].pos_ in ("ADP", "CCONJ", "SCONJ", "PART")
            or toks[-1].pos_ in ("ADP", "AUX", "CCONJ", "SCONJ", "PART", "DET"))


def check_atom(pred, toks, full, has_pair, drug_idx):
    if not toks or not has_pair(toks):
        return None
    repaired = False
    if is_fragment(pred, toks) or missing_complement(toks, drug_idx):
        if has_pair(full) and len(full) <= FALLBACK_MAX_LEN and [t.i for t in full] != [t.i for t in toks]:
            toks, repaired = full, True
        else:
            return None
    if not (MIN_LEN <= len(toks) <= (FALLBACK_MAX_LEN if repaired else MAX_LEN)):
        return None
    if dangling(toks):
        return None
    if not triggers(toks, drug_idx):
        return None
    if missing_complement(toks, drug_idx):
        return None
    return toks, repaired


def pick_version(pred, keep_toks, trim_toks, full, has_pair, drug_idx):
    tries = [keep_toks] if KEEP_CONTEXT_PHRASES else []
    if not tries or [t.i for t in trim_toks] != [t.i for t in keep_toks]:
        tries.append(trim_toks)
    for toks in tries:
        r = check_atom(pred, toks, full, has_pair, drug_idx)
        if r is not None:
            final, repaired = r
            cut = {t.i for t in keep_toks} - {t.i for t in final}
            return final, repaired, bool(cut)
    return None


def role_of(span, preds):
    for pr in preds:
        if pr is None:
            continue
        for ch in pr.children:
            if any(t.i == span.root.i for t in ch.subtree):
                return "actor" if ch.dep_ in ACTOR_DEPS else "patient"
    return "patient"


def _first_inside(ids, spans):
    return next((sp for sp in spans if {t.i for t in sp} <= ids), None)


def make_atom(toks, pred, shared, ka, kb, mentions, tier, repaired, trimmed, drug_idx):
    ids = {t.i for t in toks}
    a, b = _first_inside(ids, mentions[ka]), _first_inside(ids, mentions[kb])
    ra, rb = role_of(a, [pred, shared]), role_of(b, [pred, shared])
    if ra == "actor" and rb != "actor":
        sub, obj, role_src = a, b, "dependency"
    elif rb == "actor" and ra != "actor":
        sub, obj, role_src = b, a, "dependency"
    else:
        (sub, obj), role_src = ((a, b) if a.start < b.start else (b, a)), "text_order"
    pair_ids = {t.i for t in a} | {t.i for t in b}
    rel_toks = [t for t in toks if t.i not in pair_ids]
    edge = r"^(and|or|with|of|by|to)\s+|\s+(and|or|with|of|by|to)$"
    rel = re.sub(edge, "", " ".join(t.text.lower() for t in rel_toks)).strip()
    rel_lemma = re.sub(edge, "", " ".join(lemma(t) for t in rel_toks)).strip()
    others = sorted(k for k, sps in mentions.items() if k not in (ka, kb) and _first_inside(ids, sps) is not None)
    return {
        "sub_extract": sub.text.lower(), "rel_extract": rel, "rel_lemma": rel_lemma, "obj_extract": obj.text.lower(),
        "role_source": role_src, "hypothesis": surface(toks),
        "source_tier": tier, "repaired": repaired, "context_trimmed": trimmed,
        "trigger_words": ";".join(sorted({trigger_name(t) for t in triggers(toks, drug_idx)})),
        "other_drugs": ";".join(others),
    }


def clause_atoms(sent_doc, first, mentions, restrict, known_pair):
    out = []
    for i, (a, ka) in enumerate(first):
        for b, kb in first[i + 1:]:
            if ka == kb:
                continue
            if restrict and not (ka in known_pair and kb in known_pair):
                continue
            p = find_predicate(a, b) or find_predicate(b, a)
            if p is None or not is_finite(p):
                continue
            need = {t.i for t in a} | {t.i for t in b}
            need_all = {t.i for sp in mentions[ka] + mentions[kb] for t in sp}

            def has_pair(toks, ka=ka, kb=kb):
                ids = {t.i for t in toks}
                return _first_inside(ids, mentions[ka]) is not None and _first_inside(ids, mentions[kb]) is not None

            cands = [("clause_predicate", p, None,
                      core_clause(p, need, True), core_clause(p, need, False),
                      [t for t in p.subtree if t.pos_ != "PUNCT"])]
            if SPLIT_CONJUNCTS and any(c.dep_ == "conj" and any(t.i in need for t in c.subtree) for c in p.children):
                cands = []
            if SPLIT_CONJUNCTS:
                for c in p.conjuncts:
                    if c.pos_ in ("VERB", "AUX", "ADJ"):
                        cands.append(("conjunct_predicate", c, p,
                                      conjunct_clause(p, c, need_all, True), conjunct_clause(p, c, need_all, False),
                                      conjunct_clause(p, c, need_all, full=True)))
            for tier, pred, shared, keep_toks, trim_toks, full in cands:
                r = pick_version(pred, keep_toks, trim_toks, full, has_pair, need_all)
                if r is None:
                    continue
                toks, repaired, trimmed = r
                out.append(make_atom(toks, pred, shared, ka, kb, mentions, tier, repaired, trimmed, need_all))
    return out


# ── dataframe driver (notebook's atomize_dataframe, use_claims=False path) ───────────────────────
def _no_atom(info, reason):
    return {**info, **dict.fromkeys(ATOM_FIELDS), "source_tier": "filtered_out", "filter_reason": reason}


def atomize_dataframe(df, *, id_col="original_id", text_col="hypothesis", premise_col="premise",
                      e1_col="e1_text", e2_col="e2_text", label_col="label", split_name="", use_claims=False):
    """Decompose every row's `text_col` into atoms. use_claims=False matches the deployment
    path: the whole answer is parsed and every atom inherits the row label (unused at inference)."""
    _ensure_ready()
    records, units, stats = [], [], Counter()

    for pos, (_, row) in enumerate(df.iterrows()):
        src = int(row["source_row"]) if "source_row" in df.columns else pos
        text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
        e1 = str(row[e1_col]).strip() if pd.notna(row[e1_col]) else ""
        e2 = str(row[e2_col]).strip() if pd.notna(row[e2_col]) else ""
        resp_label = str(row[label_col]).strip().lower() if label_col in df.columns and pd.notna(row[label_col]) else "neutral"
        base = {"source_row": src, "original_id": row[id_col], "item_id": row.get("item_id", row[id_col]),
                "gen_mode": row.get("gen_mode", "normal"), "Entity1": e1, "Entity2": e2,
                "premise": row[premise_col], "source_text": text, "response_label": resp_label}
        whole = {**base, "claim_idx": -1, "claim_text": None, "label": resp_label, "label_source": "response"}

        if not text:
            records.append(_no_atom(whole, "empty_text"))
            continue
        if not e1 or not e2 or e1.lower() == e2.lower():
            records.append(_no_atom(whole, "same_or_missing_drug"))
            continue
        units.append((whole, text))

    seen = {}
    for (info, _utext), doc in zip(units, nlp.pipe((u[1] for u in units), batch_size=256)):
        lab = info["label"] if info["label"] in LABELS else "neutral"
        known_pair = {info["Entity1"].lower(), info["Entity2"].lower()}
        restrict = NEUTRAL_SAME_PAIR or lab != "neutral"
        n_before = len(records)
        for sentence in doc.sents:
            sent_doc = sentence.as_doc()
            first, mentions = entity_spans(sent_doc, known_pair)
            if len(first) < 2:
                continue
            for atom in clause_atoms(sent_doc, first, mentions, restrict, known_pair):
                key = (info["source_row"], atom["hypothesis"])
                if key in seen:
                    continue
                seen[key] = lab
                records.append({**info, **atom, "filter_reason": None})
        if len(records) == n_before:
            records.append(_no_atom(info, "no_atom"))

    out = pd.DataFrame(records, columns=OUT_COLUMNS)
    out = out.sort_values(["source_row", "claim_idx"], kind="stable").reset_index(drop=True)
    out.attrs["stats"] = dict(stats)
    return out


def atomize_text(text: str, drug1: str, drug2: str) -> pd.DataFrame:
    """Decompose one free-text answer about (drug1, drug2) into atomic claims.

    Returns only the kept atoms (source_tier != 'filtered_out'), one row per atom,
    with columns including sub_extract, rel_extract, obj_extract and hypothesis
    (the atom's surface text — what the NLI model reads as the hypothesis)."""
    df = pd.DataFrame([{
        "source_row": 0, "original_id": "user-0", "premise": "", "hypothesis": text,
        "e1_text": drug1, "e2_text": drug2, "label": "neutral",
    }])
    out = atomize_dataframe(df, use_claims=False)
    return out[out["source_tier"] != "filtered_out"].reset_index(drop=True)
