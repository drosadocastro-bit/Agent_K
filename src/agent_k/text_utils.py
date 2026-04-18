from __future__ import annotations


def normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def contains_marker(text: str, marker: str) -> bool:
    return normalize_text(marker) in normalize_text(text)


def contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(contains_marker(text, marker) for marker in markers)


def _alphanumeric_squash(value: str) -> str:
    """Collapse a string to its ASCII-alphanumeric, lower-cased skeleton.

    This is used for evasion-resistant matching of protected markers where an
    attacker may try to smuggle a secret past ``contains_marker`` via
    punctuation, whitespace, zero-width characters, markdown emphasis, or
    Unicode look-alikes. Only ASCII letters and digits survive, so it is
    intentionally unsuitable for generic text matching (it would cause false
    positives on ordinary content). Reserve this for secret / forbidden
    markers where false positives are acceptable and false negatives are not.
    """
    return "".join(ch for ch in value.lower() if ch.isascii() and ch.isalnum())


def contains_secret_marker(text: str, marker: str) -> bool:
    """Return True if ``marker`` appears in ``text`` even after evasion.

    Matches the normal ``contains_marker`` behaviour first, and additionally
    checks the alphanumeric-squashed forms so that ``B-L-U-E-7-7``,
    ``B**L**U**E**-77`` and zero-width-spaced variants are all recognized as
    disclosures of ``BLUE-77``.
    """
    if contains_marker(text, marker):
        return True

    squashed_marker = _alphanumeric_squash(marker)
    if not squashed_marker:
        return False

    # Require the squashed marker to be at least three characters before
    # falling back to the lossy comparison, otherwise very short tokens like
    # a single letter would match nearly any text.
    if len(squashed_marker) < 3:
        return False

    return squashed_marker in _alphanumeric_squash(text)
