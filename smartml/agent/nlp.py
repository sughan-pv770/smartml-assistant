"""
smartml.agent.nlp
=================
Spoken-English normalisation, compound command splitting, safe dataset file
discovery with directory boundary validation, and concise speech summarisation.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Sequence

ALLOWED_DATA_EXTENSIONS = {
    ".csv", ".tsv", ".xlsx", ".xls", ".json", ".parquet", ".feather",
}

_NUMBER_WORDS: dict[str, str] = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
}

_STOPWORDS = {
    "load", "open", "read", "use", "file", "dataset", "data", "table",
    "the", "my", "a", "an", "from", "in", "on", "please", "can", "you",
    "for", "me", "called", "named", "csv", "xlsx", "xls", "json", "parquet",
}


def normalize_transcript(text: str) -> str:
    """Normalise spoken English transcript into clean command text.

    - Converts 'dot csv' / 'point csv' -> '.csv'
    - Converts 'underscore' -> '_'
    - Converts standalone number words ('three' -> '3')
    """
    if not text:
        return ""
    out = text.strip()
    # Spoken file extensions
    out = re.sub(
        r"\s+(?:dot|point)\s+(csv|tsv|xlsx|xls|json|parquet|feather|joblib)\b",
        r".\1",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(r"\s+underscore\s+", "_", out, flags=re.IGNORECASE)

    # Replace standalone number words
    tokens = out.split()
    converted: list[str] = []
    for tok in tokens:
        clean_tok = re.sub(r"[^a-zA-Z]", "", tok).lower()
        if clean_tok in _NUMBER_WORDS and tok.isalpha():
            converted.append(_NUMBER_WORDS[clean_tok])
        else:
            converted.append(tok)
    return " ".join(converted)


def split_compound_commands(text: str) -> list[str]:
    """Split a compound spoken request into ordered single-intent commands.

    Preserves 'and' inside phrases like 'between X and Y' or 'action=X and reward=Y'.
    """
    norm = normalize_transcript(text)
    if not norm:
        return []

    # Split on explicit sequential connectors first
    parts = re.split(
        r"\b(?:and\s+then|then\s+also|after\s+that|followed\s+by|;\s*)\b",
        norm,
        flags=re.IGNORECASE,
    )

    action_verbs = (
        r"(?:load|open|clean|preprocess|analyze|analyse|summarize|summarise|"
        r"engineer|select|train|build|fit|cluster|segment|group|detect|remove|"
        r"drop|rename|fill|replace|filter|tune|forecast|plot|visualize|visualise|"
        r"generate|show|save|predict|explain|why|set\s+target)"
    )
    refined: list[str] = []
    for part in parts:
        # Split on " and <action_verb>" when not preceded by "between"
        subparts = re.split(
            rf"(?<!between\s)\band\s+(?={action_verbs}\b)",
            part.strip(),
            flags=re.IGNORECASE,
        )
        for sp in subparts:
            cleaned = sp.strip(" ,.")
            if cleaned:
                refined.append(cleaned)
    return refined


def is_within_allowed_dirs(
    candidate: Path,
    allowed_dirs: Sequence[str | Path],
) -> bool:
    """Return True only if *candidate* resolves inside one of *allowed_dirs*.

    Prevents both '..' path traversal and symlink escapes.
    """
    try:
        resolved_candidate = Path(candidate).resolve(strict=True)
    except OSError:
        return False

    for folder in allowed_dirs:
        try:
            resolved_folder = Path(folder).resolve()
            if resolved_candidate.is_relative_to(resolved_folder):
                return True
        except (OSError, ValueError):
            continue
    return False


def find_dataset_candidates(
    query: str,
    search_dirs: Optional[Sequence[str | Path]] = None,
    allow_outside: bool = False,
    max_depth: int = 2,
    max_files: int = 500,
) -> list[str]:
    """Find dataset files matching *query* (exact path or partial spoken name).

    Security guarantee:
        Unless *allow_outside* is True (used only when the user interactively
        selects a file in the native OS file dialog subprocess), any explicit
        path or discovered file MUST resolve inside *search_dirs* via
        ``Path.resolve().is_relative_to()``.

    Returns:
        Sorted list of matching absolute file paths.
    """
    if not query or not query.strip():
        return []

    if search_dirs is None:
        home = Path.home()
        search_dirs = [
            Path.cwd(),
            home / "Downloads",
            home / "Desktop",
            home / "Documents",
        ]
    allowed_paths = [Path(d) for d in search_dirs if d is not None]

    norm_q = normalize_transcript(query).strip()

    # 1. Check if the query contains or is an explicit path
    explicit_candidates: list[Path] = []
    raw_stripped = norm_q.strip("\"' ")
    if os.path.splitext(raw_stripped)[1].lower() in ALLOWED_DATA_EXTENSIONS:
        explicit_candidates.append(Path(raw_stripped))

    path_match = re.search(
        r"([A-Za-z]:[\\/][\w\-./\\ ]+\.(?:csv|tsv|xlsx?|json|parquet|feather)|"
        r"[\w\-./\\]+\.(?:csv|tsv|xlsx?|json|parquet|feather))",
        norm_q,
        re.IGNORECASE,
    )
    if path_match:
        explicit_candidates.append(Path(path_match.group(1).strip()))

    for cand in explicit_candidates:
        # Try direct path first, then relative to each allowed directory
        probe_list = [cand] + [d / cand for d in allowed_paths]
        for probe in probe_list:
            try:
                if probe.is_file() and probe.suffix.lower() in ALLOWED_DATA_EXTENSIONS:
                    if allow_outside or is_within_allowed_dirs(probe, allowed_paths):
                        return [str(probe.resolve())]
                    else:
                        # Explicitly reject path outside allowed folders
                        return []
            except OSError:
                continue

    # 2. Extract meaningful search tokens from spoken query (e.g. "load the student file" -> {"student"})
    words = re.findall(r"[a-z0-9]+", norm_q.lower())
    query_tokens = [w for w in words if w not in _STOPWORDS and len(w) >= 2]
    if not query_tokens:
        return []

    # 3. Walk allowed directories (max depth 2 and capped file count) and score files by token overlap
    scored_matches: list[tuple[int, int, str]] = []
    seen_resolved: set[str] = set()
    scanned_files = 0

    for base_dir in allowed_paths:
        if scanned_files >= max_files:
            break
        try:
            resolved_base = base_dir.resolve()
            if not resolved_base.is_dir():
                continue
        except OSError:
            continue

        for root, dirs, files in os.walk(resolved_base):
            rel_depth = len(Path(root).relative_to(resolved_base).parts)
            if rel_depth > max_depth:
                dirs[:] = []
                continue
            if rel_depth >= max_depth:
                dirs[:] = []
            # Skip hidden/venv directories
            dirs[:] = [
                d for d in dirs
                if not d.startswith(".") and d not in ("venv", ".venv", "node_modules", "__pycache__", "build", "dist")
            ]
            for fname in files:
                if scanned_files >= max_files:
                    dirs[:] = []
                    break
                ext = os.path.splitext(fname)[1].lower()
                if ext not in ALLOWED_DATA_EXTENSIONS:
                    continue
                scanned_files += 1
                fpath = Path(root) / fname
                if not allow_outside and not is_within_allowed_dirs(fpath, allowed_paths):
                    continue
                try:
                    resolved_str = str(fpath.resolve())
                except OSError:
                    continue
                if resolved_str in seen_resolved:
                    continue
                seen_resolved.add(resolved_str)

                stem_lower = os.path.splitext(fname)[0].lower()
                file_tokens = set(re.findall(r"[a-z0-9]+", stem_lower))

                # Count how many query tokens match a file token (exact or prefix/substring)
                matched_count = 0
                for qt in query_tokens:
                    if qt in file_tokens or qt in stem_lower:
                        matched_count += 1

                if matched_count > 0:
                    # Rank by: (1) number of matched query tokens descending,
                    # (2) exact stem equality, (3) shorter filename
                    exact_bonus = 1 if stem_lower == "_".join(query_tokens) or stem_lower == "".join(query_tokens) else 0
                    scored_matches.append((matched_count, exact_bonus, resolved_str))

    if not scored_matches:
        return []

    max_token_hits = max(m[0] for m in scored_matches)
    top_hits = [m for m in scored_matches if m[0] == max_token_hits]
    # If one file is an exact stem match among top hits, return it uniquely
    exact_hits = [m[2] for m in top_hits if m[1] == 1]
    if len(exact_hits) == 1:
        return exact_hits

    return sorted(m[2] for m in top_hits)


def find_dataset_file(
    query: str,
    search_dirs: Optional[Sequence[str | Path]] = None,
    allow_outside: bool = False,
) -> Optional[str]:
    """Return a single best matching dataset file path, or None if 0 matches."""
    candidates = find_dataset_candidates(query, search_dirs=search_dirs, allow_outside=allow_outside)
    if len(candidates) == 1:
        return candidates[0]
    return candidates[0] if candidates else None


def speakable(text: str, max_chars: int = 280) -> str:
    """Condense a multi-line Assistant reply into concise, natural spoken English."""
    if not text:
        return ""
    # Strip markdown bullets, checkboxes, and arrows
    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Skip raw leaderboard rows except the winner or summary lines
        if "CV=" in line and "← best" not in line:
            continue
        line = re.sub(r"^(\[x\]|\[\s\]|[•\-*])\s*", "", line)
        line = line.replace("← best", "(winner)")
        cleaned_lines.append(line)

    joined = ". ".join(cleaned_lines)
    # Replace multiplication sign '×' with spoken 'by' and strip emojis for TTS
    joined = joined.replace("×", "by")
    joined = re.sub(r"[\U00010000-\U0010ffff]", "", joined)
    joined = re.sub(r"\.\.+", ".", joined)
    joined = re.sub(r"\s+", " ", joined).strip()

    if len(joined) <= max_chars:
        return joined

    # Truncate cleanly at a sentence or word boundary
    cut = joined[: max_chars - 30].rsplit(" ", 1)[0].rstrip(".,;:")
    return f"{cut}. Full details are on screen."

