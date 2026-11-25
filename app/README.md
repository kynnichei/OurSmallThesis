# DDI Claim-Verification Demo

Interactive demo of the drug–drug interaction verification framework. Pick two
drugs from the DDI-2013 corpus, generate a medical answer with Gemini (or paste your
own), and the app:

1. looks up the gold DDI-2013 sentence for the pair (the **premise**) and gives it to
   **Gemini** as a reference passage (simple RAG); Gemini writes a 4-sentence answer,
2. decomposes the answer into **atomic SVO claims** (scispaCy dependency parser),
3. classifies each claim against the same premise with the fine-tuned
   `atomic_pubmedbert` NLI model (entailment / contradiction / neutral),
4. rolls the per-claim verdicts into an **overall verdict + FActScore** (fraction of
   claims the reference supports).

Full flow: user picks drugs → premise lookup → Gemini answers from it → atomise → NLI verify → score.
You can also skip Gemini and paste any answer text to verify directly.

## Files

| File | Role |
|---|---|
| `app.py` | Gradio UI (the four thesis views) |
| `gemini.py` | Gemini call — generates the medical answer to verify |
| `pipeline.py` | `verify()` — atomise → NLI predict → aggregate |
| `atomiser.py` | SVO decomposition, lifted verbatim from `2_Atomisation_patched.ipynb` |
| `premise_lookup.py` | drug pair → gold DDI-2013 premise; corpus drug list for the dropdowns |
| `requirements.txt` | deployment dependencies |

The NLI model is loaded from `../atomic_pubmedbert/`. The drug list and premises are
built from `../{train,val,test}_raw_manifest.csv`. Gemini needs `GEMINI_API_KEY` in
`../.env` (local) or a Space secret (deployed); without it the Gemini button is hidden
and you can still paste an answer.

## Run locally

A dedicated virtual environment (`.venv-app/`) is used so the notebook environment is
left untouched (scispaCy pins older numpy/scipy).

```powershell
# from the repo root
py -3.12 -m venv .venv-app
.\.venv-app\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv-app\Scripts\python.exe -m pip install transformers==5.15.0 scispacy==0.6.2 "gradio>=4.44" pandas==3.0.3 google-genai==2.3.0 python-dotenv==1.1.1
.\.venv-app\Scripts\python.exe -m pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz

# launch (serves http://127.0.0.1:7860)
.\.venv-app\Scripts\python.exe -m app.app
```

Run it as a module from the repo root (`-m app.app`), not `python app/app.py`, so the
package-relative imports resolve.

## Notes

- Only interacting pairs are offered: pairs with `interaction_type == "none"` are dropped,
  because their sentence does not state an interaction. Drug 1 lists drugs with at least
  3 interacting pairs (742 drugs; rare spelling variants such as "R(+) warfarin" drop out).
  Drug 2 lists every interacting partner of Drug 1.
- Each pair uses the **first** matching sentence in train → val → test order, so most
  premises come from the training data. Fine for a demo; not an evaluation.
- Gemini's format rules copy notebook 1's style (both drugs named in every sentence, clear
  effect/advice verbs, no "because"/"due to" clauses) so the rule-based atomiser can split it.
