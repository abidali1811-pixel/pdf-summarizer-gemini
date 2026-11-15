"""Prompt templates for the PDF summarizer.

One system persona, four summary styles, and four task prompts. Keeping every
instruction here means prompt work never touches application logic.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

SYSTEM = (
    "You are a precise document summarizer. Use only information found in the "
    "provided text; never invent facts, numbers, names or quotes. Write in clear, "
    "plain English. Respond with the requested content only — no preamble, no "
    "closing remarks, no meta commentary about the task."
)

STYLES: dict[str, str] = {
    "brief": (
        "A brief summary of 2-3 sentences covering the main point and the single "
        "most important takeaway."
    ),
    "detailed": (
        "A detailed summary. Cover:\n"
        "- Main topics and themes\n"
        "- Key arguments or findings\n"
        "- Important details and supporting examples\n"
        "- Conclusions or recommendations"
    ),
    "bullet_points": (
        "A scannable list of bullet points. Group related information together, "
        "use sub-bullets where it helps, and keep each bullet to one idea."
    ),
    "executive": (
        "An executive summary for a busy decision maker. Cover:\n"
        "- Key insights and findings\n"
        "- Strategic implications\n"
        "- Actionable recommendations\n"
        "- Risks, costs or open questions"
    ),
}

STYLE_LABELS: dict[str, str] = {
    "brief": "Brief",
    "detailed": "Detailed",
    "bullet_points": "Bullet points",
    "executive": "Executive",
}


def _template(human: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", human)])


CHUNK_SUMMARY = _template(
    "Summarize the section below in this style:\n{style}\n\n"
    "The section is part of a larger document; summarize only what it contains "
    "and do not refer to missing context.\n\n"
    "<section>\n{text}\n</section>"
)

COMBINE = _template(
    "Below are summaries of consecutive sections of one document, in order. Merge "
    "them into a single cohesive summary in this style:\n{style}\n\n"
    "Remove repetition, preserve the original ordering of ideas, and do not add "
    "information that is not present.\n\n"
    "<summaries>\n{summaries}\n</summaries>"
)

ANALYSIS = _template(
    "Analyze the structure and content of the document excerpt below. Use these "
    "headings, one short paragraph or list per heading:\n\n"
    "1. Document type\n"
    "2. Main topics and themes\n"
    "3. Key sections or chapters\n"
    "4. Intended audience\n"
    "5. Overall purpose\n\n"
    "If the excerpt is only part of a document, say so in one line at the end.\n\n"
    "<document>\n{text}\n</document>"
)

QUOTES = _template(
    "Extract 5-10 quotes from the document below that best represent its main "
    "ideas. Copy each quote verbatim, joining any line-wrapped text into one "
    "line, and keep each under 40 words. Output one quote per line with no "
    "numbering, bullets or surrounding quotation marks. If there are fewer than "
    "5 quotable sentences, output the ones that exist.\n\n"
    "<document>\n{text}\n</document>"
)
