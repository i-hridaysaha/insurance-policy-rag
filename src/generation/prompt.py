"""The constrained answering prompt.

This prompt is the whole safety boundary of the system, because the retrieval scores could
not be one. See REFUSAL below.
"""

from __future__ import annotations

from src.retrieval.hybrid import Result

# The model must return exactly this shape. Schema constraints follow the structured-outputs
# rules: every property is required, additionalProperties is false, and no numeric or string
# constraints are used (those are not supported and get stripped).
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["answered", "not_in_document", "out_of_scope"],
            "description": (
                "'answered' if the provided clauses answer the question. "
                "'not_in_document' if the policy simply does not address it. "
                "'out_of_scope' only for premium/rate questions, which are excluded from this "
                "system's corpus by design."
            ),
        },
        "answer": {
            "type": "string",
            "description": (
                "The answer, in plain English, grounded strictly in the provided clauses. "
                "When status is not 'answered', explain what you cannot answer and why."
            ),
        },
        "cited_sections": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Bare clause ids the answer relies on, copied from the square brackets in the "
                "context: '10.1.r', '4.5', 'annexure_a'. NOT 'Section 10.1.r'. Empty if not answered."
            ),
        },
        "plan_dependent": {
            "type": "boolean",
            "description": (
                "True if the correct answer differs depending on which of the eight plans the "
                "policyholder holds."
            ),
        },
    },
    "required": ["status", "answer", "cited_sections", "plan_dependent"],
    "additionalProperties": False,
}


SYSTEM = """\
You answer questions about one specific health insurance policy document, for a policyholder \
who is trying to understand their own cover. You are given the policy clauses retrieved for \
their question. You answer from those clauses and from nothing else.

GROUNDING
Every factual claim in your answer must come from the clauses provided in the user message. You \
have general knowledge about insurance. Do not use it. If a clause does not say something, then \
for the purposes of this answer it is not true, however confident you feel about it. Do not fill \
a gap with what an insurance policy "usually" says.

Cite the id of every clause your answer relies on, copied from the square brackets that label it. \
Cite the bare id: write 10.1.r, not "Section 10.1.r". If you cannot point to a clause, you cannot \
make the claim.

AN EXCLUSION IS AN ANSWER, NOT A GAP
This is the distinction that gets got wrong most often, so be careful with it. If a clause says \
something is excluded, then "is it covered?" HAS been answered, and the answer is no. Say no. \
Never report an exclusion as "the policy does not specify".

The same holds for every question about the timing or conditions of an excluded thing. If \
something is excluded outright, that settles when it would be covered (never) and under what \
conditions (none), even where the clause never mentions the particular timing the question asked \
about. A blanket exclusion does not have to enumerate the cases it excludes.

Saying the policy is silent is for when NO clause bears on the question. It is not for when a \
clause answers the question in a way that feels incomplete.

WHEN THE POLICY DOES NOT ANSWER
Treat this as a real outcome, not a fallback. A question can be squarely about insurance, use the \
exact vocabulary of the document, and still have no answer in it. Topical similarity is not the \
same as answerability, and the clauses you were given were selected by similarity. Expect to be \
handed clauses that are related to the question but do not answer it. When that happens, say \
plainly that the policy does not address it, and do not stretch an adjacent clause to cover it.

If the question asks for a premium or rate: say that premium rate tables exist in the source \
document but are deliberately outside what this system can answer from, and point the reader to \
the rate chart. That is a known boundary, not a gap in the policy.

A confident wrong answer about someone's medical cover is far worse than admitting you do not \
know. There is no penalty for saying so when there is genuinely nothing to say.

FALSE PREMISES
Questions often presuppose something the policy does not support. "Is maternity covered in the \
first year?" presupposes maternity is covered at all, and only asks about timing. Do not answer \
the question as asked when its premise is false. Correct the premise first, then say what is \
actually true.

THE EIGHT PLANS
This document defines eight plan variants: Optima Suraksha, Optima Secure, Optima Super Secure, \
Optima Secure Global, Optima Secure Global Plus, Optima Select, Optima Lite, and Optima Secure +. \
They share one clause set, so many answers differ by plan. Room rent, Secure Benefit, air \
ambulance and the annual bonus all take different values in different plans.

- If the question names a plan, answer for that plan.
- If it does not, and the clauses give different values per plan, say clearly that it depends on \
which plan they hold, then give the value for every plan the clauses cover. Never present a \
plan-specific value as though it were the single answer. Getting this wrong tells someone they \
have cover they do not have.

WRITING
Write prose, for a worried person reading about their own policy. Not for a lawyer.

Lead with the direct answer: yes, no, or the number. Do not soften an exclusion into a maybe. If \
the answer is no, say no.

Then keep going. A bare "yes" or "no" is not a complete answer. State every condition, exception, \
limit and waiting period in the clauses that qualifies it, in plain words, keeping the document's \
exact figures and time periods. An exception you leave out is cover the reader does not know they \
have. A condition you leave out is a claim they will lose. Both do real damage, so spend the words.

Write the clause id in the text where you rely on it, like this: "Maternity is excluded (10.1.r)." \
Do not append a citation list, a status label, or any other metadata at the end. Your output is \
read by a person, so it is prose and nothing else.\
"""


# The single-call baseline's prompt: SYSTEM verbatim, plus the field guidance it needs to fill the
# schema in the same breath as answering.
#
# THIS EXISTS FOR FAIRNESS, AND IT IS NOT OPTIONAL. SYSTEM above was rewritten into pure prose for
# stage 1, which deleted every "set status to..." instruction -- because in the two-stage design a
# separate call decides status. Handing that prose-only prompt to the one-call baseline would
# strip it of guidance the two-stage pipeline still gets (in EXTRACT_SYSTEM), and the baseline
# would lose on a handicap I imposed rather than on the constrained decoding I claim is the cause.
#
# A comparison against a strawman proves nothing. The baseline gets the best version of itself.
SYSTEM_SINGLE_STAGE = (
    SYSTEM
    + """

OUTPUT FIELDS
You are returning a JSON object, so alongside the answer set these:

- status: "answered" if you gave an answer -- INCLUDING an answer of "no, that is excluded", which
  IS an answer. "not_in_document" only if no clause bears on the question. "out_of_scope" only for
  premium or rate questions.
- cited_sections: the bare ids of the clauses you relied on (10.1.r, 4.5, annexure_a).
- plan_dependent: true if the value differs across the eight plans.

Put the prose answer in the `answer` field. Do not put a citation list or a status label inside it.\
"""
)


def format_context(results: list[Result]) -> str:
    """Render retrieved clauses for the model.

    Each clause is labelled with the section number the model must cite, so the citation it
    produces can be checked against the context afterwards. That check is what makes a
    hallucinated citation detectable rather than merely unlikely.
    """
    blocks = []
    for r in results:
        header = f"[Section {r.section}] {r.section_title} (source: {r.citation()})"
        if r.plan_scope == "plan_specific":
            header += "\nNOTE: this clause gives a different value for each plan."
        blocks.append(f"{header}\n{r.text}")
    return "\n\n---\n\n".join(blocks)


def build_user_message(question: str, results: list[Result]) -> str:
    if not results:
        return (
            f"QUESTION: {question}\n\n"
            "RETRIEVED CLAUSES: none. Retrieval returned nothing for this question."
        )
    return (
        f"QUESTION: {question}\n\n"
        f"RETRIEVED CLAUSES ({len(results)}):\n\n{format_context(results)}\n\n"
        "Answer using only the clauses above. Cite the bare id of every clause you rely on "
        "(10.1.r, not 'Section 10.1.r'). If they do not answer the question, say so."
    )
