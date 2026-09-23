"""Small, deterministic retrieval layer over approved repository documentation."""

from __future__ import annotations

import re
from pathlib import Path

KNOWLEDGE_FILES = (
    "docs/MODEL_CARD.md",
    "docs/DATA_CARD.md",
    "docs/ARCHITECTURE.md",
    "research/README.md",
)

STOP_WORDS = {
    "a",
    "about",
    "and",
    "are",
    "hai",
    "how",
    "is",
    "ka",
    "ke",
    "ki",
    "kya",
    "me",
    "model",
    "of",
    "the",
    "this",
    "to",
    "was",
}

TOKEN_ALIASES = {
    "evaluation": "validation",
    "validate": "validation",
    "validated": "validation",
    "validating": "validation",
    "limitations": "limitation",
    "thresholds": "threshold",
}


def _tokens(text: str) -> set[str]:
    return {
        TOKEN_ALIASES.get(token, token)
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in STOP_WORDS
    }


def _read_sections(path: Path) -> list[dict[str, str]]:
    sections: list[dict[str, str]] = []
    heading = "Overview"
    body: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            if body:
                sections.append(
                    {"source": path.as_posix(), "section": heading, "text": "\n".join(body).strip()}
                )
            heading = line.removeprefix("## ").strip()
            body = []
        elif not line.startswith("# "):
            body.append(line)
    if body:
        sections.append(
            {"source": path.as_posix(), "section": heading, "text": "\n".join(body).strip()}
        )
    return [section for section in sections if section["text"]]


def load_knowledge(base_path: str | Path | None = None) -> list[dict[str, str]]:
    """Load only the explicitly approved documentation files."""

    root = Path(base_path) if base_path else Path(__file__).resolve().parents[2]
    chunks: list[dict[str, str]] = []
    for relative_path in KNOWLEDGE_FILES:
        path = root / relative_path
        if path.is_file():
            for section in _read_sections(path):
                section["source"] = relative_path
                chunks.append(section)
    return chunks


def search_knowledge(
    question: str | None,
    *,
    top_k: int = 3,
    base_path: str | Path | None = None,
) -> list[dict[str, object]]:
    """Rank Markdown sections using transparent token overlap with title weighting."""

    query = _tokens(question or "validation limitations intended use")
    ranked: list[tuple[float, dict[str, str]]] = []
    for chunk in load_knowledge(base_path):
        title_tokens = _tokens(chunk["section"])
        body_tokens = _tokens(chunk["text"])
        score = 3 * len(query & title_tokens) + len(query & body_tokens)
        if score:
            ranked.append((float(score), chunk))
    ranked.sort(key=lambda item: (-item[0], item[1]["source"], item[1]["section"]))

    evidence = []
    for score, chunk in ranked[:top_k]:
        clean_text = " ".join(chunk["text"].split())
        evidence.append(
            {
                "source": chunk["source"],
                "section": chunk["section"],
                "score": score,
                "excerpt": clean_text[:500],
            }
        )
    return evidence
