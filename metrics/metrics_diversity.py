from __future__ import annotations
from typing import List, Sequence, Tuple

def _whitespace_tokens(text: str) -> List[str]:
    """Tokenize with whitespace split only (no normalization)."""
    return (text or "").split()


def compute_catalog_diversity(list_of_recommendations: Sequence[str], catalog_size: int) -> float:
    """
    Catalog diversity: (# unique recommended tracks) / (catalog size).
    """
    if catalog_size <= 0:
        return 0.0
    return len(set(list_of_recommendations)) / float(catalog_size)


def compute_lexical_diversity(list_of_responses: Sequence[str]) -> float:
    """
    Lexical diversity (TTR, micro): (# unique tokens) / (# total tokens).
    """
    unique_vocab = set()
    total_tokens = 0

    for response in list_of_responses:
        tokens = _whitespace_tokens(response)
        unique_vocab.update(tokens)
        total_tokens += len(tokens)

    if total_tokens == 0:
        return 0.0
    return len(unique_vocab) / float(total_tokens)
