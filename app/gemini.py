"""Gemini step: the RAG assistant whose answer the framework verifies.

Simple retrieval-augmented generation: the app retrieves the gold DDI-2013 sentence
for the chosen drug pair (premise_lookup.get_premise) and gives it to Gemini as the
reference passage. Gemini answers from it; the pipeline then checks every atomic
claim of the answer against that same passage. This matches how the NLI model was
trained (notebook 1 also wrote the hypotheses from the premise).

It is deliberately NOT the notebook's training-data SYSTEM_PROMPT (which writes
labelled entailment/contradiction/neutral sentences). The format rules below copy
notebook 1's style (4 sentences, both drugs named in each, clear effect/advice verbs)
so the rule-based atomiser can split the answer.

Client setup, key loading and model config are reused from 1_AugmentResponses.ipynb:
  - GEMINI_API_KEY from .env (or the process environment / a Space secret)
  - model gemini-3.5-flash-lite, temperature 0.4, top_p 0.95, top_k 40
"""
from __future__ import annotations

import os
import time
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

# Same model/sampling as the notebook's generation step.
MODEL_NAME = "gemini-3.5-flash-lite"
TEMPERATURE = 0.4
TOP_P = 0.95
TOP_K = 40
MAX_OUTPUT_TOKENS = 1024  # a short answer; the training run used a far larger budget for batched JSON
RETRIES = 3               # tries on a temporary server error (503 "high demand")

SYSTEM_PROMPT = (
    "You are a clinical drug-information assistant. Given Drug 1, Drug 2 and a reference passage "
    "retrieved from the drug-interaction literature, answer how the two drugs interact when taken together. "
    "Base the answer on the reference passage.\n"
    "Format rules (the answer is parsed by a rule-based parser, so follow them exactly):\n"
    "- Exactly 4 sentences, 40-80 words in total, plain clinical register.\n"
    "- EVERY sentence names BOTH drugs, spelled exactly as given. Never use pronouns, plurals, "
    "abbreviations or phrases like 'this combination', 'both agents' or 'this exposure' instead of the names.\n"
    "- One checkable claim per sentence, built on a clear effect, mechanism or advice verb "
    "(for example increase, decrease, reduce, enhance, inhibit, prolong, raise, lower, avoid, monitor, require).\n"
    "- Short, direct sentences: subject, verb, object. No 'because', 'due to', 'since', 'while' or 'which' clauses.\n"
    "- Never mention 'the passage', 'the reference' or 'the text'. No citations, headings, lists, or disclaimers."
)

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "Missing GEMINI_API_KEY. Put it in .env (local) or the Space secret (deployed)."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def generate_answer(drug1: str, drug2: str, premise: Optional[str] = None) -> str:
    """Ask Gemini for a short answer about (drug1, drug2), grounded in `premise` when given."""
    client = _get_client()
    prompt = f"Drug 1: {drug1}\nDrug 2: {drug2}\n"
    if premise:
        prompt += f"Reference passage: {premise}\n"
    prompt += "\nWrite the interaction answer."
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    for attempt in range(1, RETRIES + 1):
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=prompt, config=config)
            return (response.text or "").strip()
        except errors.ServerError:
            if attempt == RETRIES:
                raise
            time.sleep(2 * attempt)
