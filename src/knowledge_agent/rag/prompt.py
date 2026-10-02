"""Prompt template and context assembly for the answer pipeline."""

import re
from collections.abc import Sequence

from knowledge_agent.schemas.retrieval import SearchResult

SYSTEM_PROMPT = """
You are a technical research assistant.

Answer the user's question using only the
provided context.

If the context does not contain enough
information, clearly say so.

Do not invent facts or citations.

Context:
{context}

Question:
{question}
"""

CITATION_INSTRUCTION = (
    "Cite your sources with bracketed numbers taken from the context above, "
    "for example [1] or [1][2]. Only cite the numbers you actually used."
)

_PLACEHOLDER = re.compile(r"\{(context|question)\}")
_MARKER = re.compile(r"\[([\d\s,]+)\]")


def render_prompt(
    context: str, question: str, cite_instruction: bool = True
) -> str:
    """Fill the template in a single pass.

    ``str.format`` cannot be used here: the context is raw text lifted from
    LaTeX-heavy PDFs, so a chunk containing ``{`` or ``\\frac{a}{b}`` raises
    ``KeyError`` or ``IndexError``. Substituting with a function also means
    inserted content is never rescanned, so a chunk that literally contains
    ``{question}`` is left alone instead of being substituted a second time.

    ``cite_instruction`` is appended to the question rather than baked into
    ``SYSTEM_PROMPT``: the small local models do not infer bracketed citations
    on their own, so without it every citation reports ``cited=False``. It
    lives outside the template so the system prompt stays exactly as specified.
    """
    values = {"context": context, "question": question}

    if cite_instruction:
        values["question"] = f"{question}\n\n{CITATION_INSTRUCTION}"

    return _PLACEHOLDER.sub(lambda match: values[match.group(1)], SYSTEM_PROMPT)


def format_context(results: Sequence[SearchResult]) -> str:
    """Render retrieved chunks as a numbered, attributed block.

    The model needs the attribution to attribute its own claims, but the
    filename and page number reported to the caller come from
    ``SearchResult``, never from this text, so a hallucinated page number
    cannot reach the output.
    """
    return "\n\n".join(
        f"[{index}] ({result.filename}, page {result.page_number})\n"
        f"{result.content.strip()}"
        for index, result in enumerate(results, start=1)
    )


def trim_context(results: Sequence[SearchResult], max_chars: int) -> list[SearchResult]:
    """Drop the lowest-ranked chunks until the block fits the budget."""
    kept = list(results)

    while len(kept) > 1 and len(format_context(kept)) > max_chars:
        kept.pop()

    return kept


def cited_indexes(answer: str) -> set[int]:
    """Which ``[n]`` markers the answer actually used.

    Accepts ``[1]``, ``[1,2]`` and ``[1, 2]`` so a model that cites a range in
    one bracket is not silently credited with nothing.
    """
    found: set[int] = set()

    for match in _MARKER.finditer(answer):
        for part in match.group(1).split(","):
            token = part.strip()

            if token.isdigit():
                found.add(int(token))

    return found
