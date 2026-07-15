"""Stage 2 of the OPTIONAL two-stage path: label a free-text answer with its structure.

READ THIS BEFORE ASSUMING TWO-STAGE IS THE GOOD PATH. IT IS NOT THE DEFAULT, AND IT MEASURED WORSE.

The two-stage design was a hypothesis: reason in free text (stage 1), then extract structure under
grammar constraint (stage 2), on the theory that constraining the answer itself degraded the
model's reading. Measured across 34 questions against a fair one-stage baseline, the hypothesis
failed -- one-stage verified 100% of its citations, two-stage only 61.8%, because the free-text
prose in stage 1 reaches for fine-grained sub-clause ids ("4.6.iv.b", "10.1.q.i") that the system
cannot verify at its section-level index, and this extractor faithfully transcribes them.

So this module is kept only to make `scripts/evaluate.py --stages` reproducible, and the default
in Answerer is one-stage. See the long comment on Answerer for the full result. The extractor's
one firm rule -- copy ids verbatim, never invent or correct -- is what lets those unverifiable
sub-clause references survive to the citation guard and be counted against two-stage honestly,
rather than being quietly cleaned up.
"""

from __future__ import annotations

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["answered", "not_in_document", "out_of_scope"],
            "description": (
                "What the answer did. 'answered' if it gave an answer -- INCLUDING an answer of "
                "'no, that is excluded', which is an answer. 'not_in_document' only if it said the "
                "policy does not address the question. 'out_of_scope' only if it said premium/rate "
                "questions are outside this system."
            ),
        },
        "cited_sections": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Every clause id the answer refers to, as bare ids: '10.1.r', '4.5', 'annexure_a'. "
                "Copy only ids that literally appear in the answer text. Do not add, infer or "
                "correct any. Empty list if it cites none."
            ),
        },
        "plan_dependent": {
            "type": "boolean",
            "description": "True if the answer says the value differs between the eight plans.",
        },
    },
    "required": ["status", "cited_sections", "plan_dependent"],
    "additionalProperties": False,
}


EXTRACT_SYSTEM = """\
You are labelling an answer that has already been written. You are not answering anything, not \
improving anything, and not checking anything for correctness.

Read the answer text and record three facts about it:

1. status  -- what did the answer DO?
   - "answered" if it gave the policyholder an answer. An answer of "no, that is excluded" is an
     ANSWER. So is "yes, but only under these conditions". Do not mark those as not_in_document.
   - "not_in_document" ONLY if the answer explicitly says the policy does not address the question.
   - "out_of_scope" ONLY if the answer says premium or rate questions fall outside this system.

2. cited_sections -- which clause ids does the answer text mention? Copy them exactly as bare ids
   (10.1.r, 4.5, annexure_a). Copy ONLY ids that literally appear in the text. Never add one you
   think it should have cited, and never correct one you think is wrong. You are transcribing, not
   judging: an invented id must survive to the output so it can be caught downstream.

3. plan_dependent -- does the answer say the value differs across the eight plans?

Report what the answer says, not what it should have said.\
"""


def build_extract_message(question: str, answer_text: str) -> str:
    return (
        f"QUESTION THAT WAS ASKED\n{question}\n\n"
        f"ANSWER THAT WAS WRITTEN\n{answer_text}\n\n"
        "Label this answer: status, the clause ids it mentions, and whether it is plan-dependent."
    )
