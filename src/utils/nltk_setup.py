from __future__ import annotations

import nltk
from nltk.data import find
from typing import Iterable
from utils.logging import get_logger

LOGGER = get_logger(__name__)

DEFAULT_RESOURCES: list[tuple[str, str]] = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),   # required by NLTK >= 3.8
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
]
_ensured = False   # idempotency guard — only runs once per session

def ensure_nltk(resources: Iterable[tuple[str, str]] = DEFAULT_RESOURCES) -> None:
    """Ensure required NLTK resources are available. Downloads missing ones."""
    global _ensured
    if _ensured:
        return
    for path, pkg in resources:
        try:
            find(path)
        except LookupError:
            LOGGER.info("Downloading NLTK package: %s", pkg)
            nltk.download(pkg, quiet=True)
    _ensured = True