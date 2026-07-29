"""Deterministic lexical checks for counterfactual token replacements."""

from __future__ import annotations

from functools import lru_cache
from threading import Lock


_ENTITY_TYPES = {
    "PERSON",
    "NORP",
    "FAC",
    "ORG",
    "GPE",
    "LOC",
    "PRODUCT",
    "EVENT",
    "WORK_OF_ART",
    "LAW",
    "LANGUAGE",
    "DATE",
    "TIME",
}
_POS_MAP = {
    "NOUN": "n",
    "VERB": "v",
    "ADJ": "a",
    "ADV": "r",
}
_WORDNET_LOCK = Lock()


def ensure_wordnet_available() -> None:
    """Fail early instead of silently disabling the anti-synonym contract."""

    try:
        from nltk.corpus import wordnet

        with _WORDNET_LOCK:
            wordnet.synsets("entity", pos="n")
    except (ImportError, LookupError) as error:
        raise RuntimeError(
            "WordNet is required for replacement-pool construction; install "
            "nltk and download the wordnet corpus"
        ) from error


@lru_cache(maxsize=32768)
def is_lexical_paraphrase(
    original: str,
    candidate: str,
    pos: str,
    unit_type: str,
) -> bool:
    """Detect synonyms and close lexical paraphrases with WordNet."""

    if str(unit_type).upper() in _ENTITY_TYPES:
        return False
    left = str(original).strip().casefold()
    right = str(candidate).strip().casefold()
    if not left or not right or left == right:
        return True
    wordnet_pos = _POS_MAP.get(str(pos).upper())
    if wordnet_pos is None:
        return False
    try:
        from nltk.corpus import wordnet
    except ImportError as error:
        raise RuntimeError("nltk is required for lexical validation") from error
    # NLTK's lazy ZipFile-backed corpus reader is not thread-safe.
    with _WORDNET_LOCK:
        try:
            left_lemma = wordnet.morphy(left, wordnet_pos) or left
            right_lemma = wordnet.morphy(right, wordnet_pos) or right
            if left_lemma == right_lemma:
                return True
            left_synsets = set(wordnet.synsets(left_lemma, pos=wordnet_pos))
            right_synsets = set(wordnet.synsets(right_lemma, pos=wordnet_pos))
        except LookupError as error:
            raise RuntimeError(
                "the NLTK WordNet corpus is required for lexical validation"
            ) from error
        if left_synsets & right_synsets:
            return True
        for left_synset in left_synsets:
            for right_synset in right_synsets:
                similarity = left_synset.path_similarity(right_synset)
                if similarity is not None and similarity >= 0.5:
                    return True
                wup_similarity = left_synset.wup_similarity(right_synset)
                if wup_similarity is not None and wup_similarity >= 0.85:
                    return True
    return False


@lru_cache(maxsize=32768)
def lexical_paraphrase_candidates(
    original: str,
    pos: str,
    *,
    limit: int = 8,
) -> tuple[str, ...]:
    """Return direct WordNet synonyms under the shared corpus lock."""

    wordnet_pos = _POS_MAP.get(str(pos).upper())
    if wordnet_pos is None:
        return ()
    try:
        from nltk.corpus import wordnet
    except ImportError:
        return ()
    source = str(original).strip()
    suggestions = []
    seen = {source.casefold()}
    with _WORDNET_LOCK:
        try:
            synsets = wordnet.synsets(source.casefold(), pos=wordnet_pos)
            for synset in synsets:
                for lemma in synset.lemma_names():
                    candidate = lemma.replace("_", " ")
                    if source[:1].isupper():
                        candidate = candidate[:1].upper() + candidate[1:]
                    folded = candidate.casefold()
                    if folded in seen or len(candidate.split()) > 5:
                        continue
                    seen.add(folded)
                    suggestions.append(candidate)
                    if len(suggestions) >= limit:
                        return tuple(suggestions)
        except LookupError:
            return ()
    return tuple(suggestions)
