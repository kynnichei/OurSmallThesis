"""Gradio demo for the DDI claim-verification framework.

Stage B (this file): the user picks two corpus drugs and pastes a medical answer
(e.g. one an LLM wrote). The app decomposes it into atomic claims, classifies each
against the gold DDI-2013 premise, and shows the four thesis views:
  - the premise (gold reference sentence)
  - the SVO decomposition (Visualisasi Dekomposisi)
  - each atom with its entailment/contradiction/neutral label (Faktualitas Atomik)
  - the aggregate verdict + FActScore-style score (Faktualitas Keseluruhan)

Run locally:  python -m app.app      (from the repo root)
The Gemini step (user picks drugs -> Gemini writes the answer) is added on top later.
"""
from __future__ import annotations

import gradio as gr

from .pipeline import verify
from .premise_lookup import drug_choices, get_premise, partners_of

try:
    from .gemini import generate_answer
    _GEMINI_READY = True
except Exception:
    # google-genai / key not available: the Gemini button is hidden, pasting still works.
    _GEMINI_READY = False

_VERDICT_EMOJI = {
    "entailment": "✅ Supported (entailment)",
    "contradiction": "❌ Contradicted (contradiction)",
    "neutral": "➖ Not stated (neutral)",
}


def _update_partners(drug1):
    """Limit Drug 2 to drugs that form a covered pair with Drug 1 (Input Terbatas)."""
    if not drug1:
        return gr.update(choices=[], value=None,
                         label="Drug 2 (pick Drug 1 first)", interactive=False)
    partners = partners_of(drug1)
    return gr.update(
        choices=partners, value=None, interactive=True,
        label=f"Drug 2 — {len(partners)} drugs interact with {drug1} in the corpus",
    )


def _run(drug1, drug2, answer):
    if not drug1 or not drug2:
        return "Pick both drugs.", "", [], [], ""
    if not answer or not answer.strip():
        return "Paste an answer to verify.", "", [], [], ""

    res = verify(answer, drug1, drug2)

    if res.note and not res.atoms:
        return res.note, res.premise, [], [], ""

    svo_rows = [[a.subject, a.relation, a.object] for a in res.atoms]
    atom_rows = [
        [a.hypothesis, _VERDICT_EMOJI[a.label], f"{a.probs[a.label]:.0%}"]
        for a in res.atoms
    ]

    counts = res.label_counts
    summary = (
        f"## {_VERDICT_EMOJI[res.verdict]}  (overall verdict)\n\n"
        f"**FActScore (supported atoms): {res.factscore:.0%}**  "
        f"— {counts.get('entailment', 0)} entailment, "
        f"{counts.get('contradiction', 0)} contradiction, "
        f"{counts.get('neutral', 0)} neutral, across {len(res.atoms)} atomic claims.\n\n"
        f"*Strict rule: any contradicted claim makes the whole answer contradicted; "
        f"otherwise any unsupported claim makes it neutral; only all-supported is entailment.*"
    )
    status = res.note or f"Verified {len(res.atoms)} atomic claims."
    return status, res.premise, svo_rows, atom_rows, summary


def _generate(drug1, drug2):
    if not drug1 or not drug2:
        return "", "Pick both drugs first.", ""
    premise = get_premise(drug1, drug2)
    if not premise:
        return "", f"No DDI-2013 reference sentence for {drug1} + {drug2}.", ""
    try:
        answer = generate_answer(drug1, drug2, premise)
    except Exception as e:
        return "", f"Gemini call failed: {e}", premise
    return answer, f"Gemini answered from the reference sentence below. Review it, then Verify.", premise


def build():
    with gr.Blocks(title="DDI Claim Verification") as demo:
        gr.Markdown(
            "# Drug–Drug Interaction Claim Verification\n"
            "Pick two drugs that interact in the DDI-2013 corpus. Gemini answers from the corpus "
            "reference sentence (or paste your own answer), then the framework decomposes the answer "
            "into atomic claims and checks each against that reference sentence."
        )

        with gr.Row():
            drug1 = gr.Dropdown(choices=drug_choices(), label="Drug 1", filterable=True)
            drug2 = gr.Dropdown(choices=[], label="Drug 2 (pick Drug 1 first)",
                                filterable=True, interactive=False)
        drug1.change(_update_partners, inputs=drug1, outputs=drug2)

        if _GEMINI_READY:
            gen_btn = gr.Button("Generate answer with Gemini")

        answer = gr.Textbox(
            label="Medical answer to verify",
            placeholder="Generate with Gemini above, or paste a few sentences about the two drugs' interaction…",
            lines=5,
        )
        run_btn = gr.Button("Verify", variant="primary")

        status = gr.Markdown()
        premise_box = gr.Textbox(label="Premise (gold DDI-2013 reference sentence)", interactive=False, lines=3)

        gr.Markdown("### Decomposition into SVO triplets")
        svo = gr.Dataframe(headers=["Subject", "Relation", "Object"], interactive=False, wrap=True)

        gr.Markdown("### Atomic factuality (per claim)")
        atoms = gr.Dataframe(headers=["Atomic claim", "Label", "Confidence"], interactive=False, wrap=True)

        gr.Markdown("### Overall factuality")
        summary = gr.Markdown()

        if _GEMINI_READY:
            gen_btn.click(_generate, inputs=[drug1, drug2], outputs=[answer, status, premise_box])

        run_btn.click(_run, inputs=[drug1, drug2, answer],
                      outputs=[status, premise_box, svo, atoms, summary])

    return demo


if __name__ == "__main__":
    build().launch()
