from __future__ import annotations

from typing import Any, Iterable


# Patterns are plain substrings matched case-insensitively after lower-casing
# each string leaf. Order is preserved so multi-match details are
# deterministic.
_SUBSTRING_PATTERNS: tuple[tuple[str, str], ...] = (
    ("path_traversal", "../"),
    ("path_traversal", "..\\"),
    ("secret_path", "/etc/shadow"),
    ("secret_path", "/etc/passwd"),
    ("secret_path", "/root/.ssh"),
    ("secret_path", "id_rsa"),
    ("secret_path", ".aws/credentials"),
    ("secret_path", ".env"),
    ("shell_metacharacter", "&&"),
    ("shell_metacharacter", "||"),
    ("shell_metacharacter", "$("),
    ("shell_metacharacter", "`"),
    ("shell_metacharacter", " | "),
    ("shell_metacharacter", "; "),
    ("shell_metacharacter", " > "),
    ("shell_metacharacter", " >> "),
    ("destructive_command", "rm -rf"),
    ("destructive_command", "mkfs"),
    ("destructive_command", "dd if="),
    ("destructive_command", "format c:"),
    ("exfiltration", "curl "),
    ("exfiltration", "wget "),
    ("exfiltration", "http://"),
    ("exfiltration", "https://"),
    ("exfiltration", "ftp://"),
)


def _flatten(value: Any) -> Iterable[str]:
    """Yield every string leaf inside a nested argument structure."""
    if value is None:
        return
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, (int, float, bool)):
        yield str(value)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                yield key
            yield from _flatten(item)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _flatten(item)
        return
    # Unknown types: stringify defensively so they still get scanned.
    yield str(value)


def scan_arguments(arguments: Any) -> tuple[tuple[str, str], ...]:
    """Return a tuple of ``(category, matched_fragment)`` pairs.

    An empty tuple means no dangerous pattern was found. The fragment is the
    literal pattern matched, lower-cased, so it is safe to embed verbatim in
    audit messages.
    """
    matches: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for leaf in _flatten(arguments):
        lowered = leaf.lower()
        for category, pattern in _SUBSTRING_PATTERNS:
            if pattern in lowered:
                key = (category, pattern)
                if key in seen:
                    continue
                seen.add(key)
                matches.append(key)

    return tuple(matches)


def describe(matches: tuple[tuple[str, str], ...]) -> str:
    """Produce a short, deterministic description for flag detail text."""
    return "; ".join(f"{category}: '{fragment}'" for category, fragment in matches)
