# -*- coding: utf-8 -*-

import re
from collections import Counter


# ============================================================
# Uncertainty expressions
# ============================================================

UNCERTAIN_PATTERN_DICT = {
    "suspicious_for": r"\bsuspicious\s+for\b",
    "suspicious_of": r"\bsuspicious\s+of\b",
    "suspicious": r"\bsuspicious\b",
    "suspected": r"\bsuspected\b",
    "suspicion_of": r"\bsuspicion\s+of\b",
    "possibly": r"\bpossibly\b",
    "possible": r"\bpossible\b",
    "probably": r"\bprobably\b",
    "probable": r"\bprobable\b",
    "r_o": r"\br\s*[/\\]\s*o\b",
    "rule_out": r"\brule\s+out\b",
    "cannot_exclude": r"\bcannot\s+exclude\b",
    "cant_exclude": r"\bcan['’]?\s*t\s+exclude\b",
    "could_not_exclude": r"\bcould\s+not\s+exclude\b",
    "suggestive_of": r"\bsuggestive\s+of\b",
    "may_represent": r"\bmay\s+represent\b",
    "may_be": r"\bmay\s+be\b",
    "concerning_for": r"\bconcerning\s+for\b",
    "concern_for": r"\bconcern\s+for\b",
}


COMPILED_UNCERTAIN_PATTERNS = {
    name: re.compile(
        pattern,
        flags=re.IGNORECASE,
    )
    for name, pattern
    in UNCERTAIN_PATTERN_DICT.items()
}


UNCERTAIN_RE = re.compile(
    "("
    + "|".join(
        f"(?:{p})"
        for p in UNCERTAIN_PATTERN_DICT.values()
    )
    + ")",
    flags=re.IGNORECASE,
)


# ============================================================
# Negative expressions
# ============================================================

NEGATIVE_PATTERN_DICT = {
    "no_evidence_of": r"\bno\s+evidence\s+of\b",
    "no_evidence_for": r"\bno\s+evidence\s+for\b",
    "negative_for": r"\bnegative\s+for\b",
    "without": r"\bwithout\b",
    "no_demonstrable": r"\bno\s+demonstrable\b",
    "no_definite": r"\bno\s+definite\b",
    "no_acute": r"\bno\s+acute\b",
    "not_seen": r"\bnot\s+seen\b",
    "not_identified": r"\bnot\s+identified\b",
    "absence_of": r"\babsence\s+of\b",
}


COMPILED_NEGATIVE_PATTERNS = {
    name: re.compile(
        pattern,
        flags=re.IGNORECASE,
    )
    for name, pattern
    in NEGATIVE_PATTERN_DICT.items()
}


NEGATIVE_RE = re.compile(
    "("
    + "|".join(
        f"(?:{p})"
        for p in NEGATIVE_PATTERN_DICT.values()
    )
    + ")",
    flags=re.IGNORECASE,
)


# ============================================================
# AIS uncertainty policy
# ============================================================

UNCERTAINTY_POLICY_TEXT = """
[AIS CODING POLICY]
Confirmed traumatic findings are positive evidence.
Suspicious, suspected, possible, possibly, probable, probably,
R/O, rule-out, cannot-exclude, suggestive, and similar uncertain
traumatic findings are also treated as candidate-positive evidence
for AIS prediction.
Explicitly negated findings are not treated as positive evidence.
""".strip()


def get_uncertainty_policy_text():
    return UNCERTAINTY_POLICY_TEXT


# ============================================================
# Detection utilities
# ============================================================

def contains_uncertain_expression(text):
    if text is None:
        return False

    return bool(
        UNCERTAIN_RE.search(
            str(text)
        )
    )


def contains_negative_expression(text):
    if text is None:
        return False

    return bool(
        NEGATIVE_RE.search(
            str(text)
        )
    )


def find_uncertain_types(text):
    if text is None:
        return []

    text = str(text)

    return sorted({
        name
        for name, pattern
        in COMPILED_UNCERTAIN_PATTERNS.items()
        if pattern.search(text)
    })


def count_uncertain_type_occurrences(text):
    if text is None:
        return {}

    text = str(text)

    out = {}

    for name, pattern in (
        COMPILED_UNCERTAIN_PATTERNS.items()
    ):
        n = len(
            pattern.findall(text)
        )

        if n:
            out[name] = n

    return out


def find_negative_types(text):
    if text is None:
        return []

    text = str(text)

    return sorted({
        name
        for name, pattern
        in COMPILED_NEGATIVE_PATTERNS.items()
        if pattern.search(text)
    })


# ============================================================
# Uncertainty normalization
# ============================================================

def normalize_uncertainty(text):
    """
    Mark non-negated uncertain findings as candidate-positive
    evidence for AIS prediction.

    The original report text is otherwise preserved.
    """

    if text is None:
        return ""

    output_lines = []

    for raw_line in str(text).splitlines():
        line = raw_line.rstrip()

        if not line.strip():
            output_lines.append(
                line
            )
            continue

        has_uncertain = bool(
            UNCERTAIN_RE.search(
                line
            )
        )

        has_negative = bool(
            NEGATIVE_RE.search(
                line
            )
        )

        if (
            has_uncertain
            and not has_negative
        ):
            if not line.lstrip().startswith(
                "[UNCERTAIN_POSITIVE]"
            ):
                line = (
                    "[UNCERTAIN_POSITIVE] "
                    + line
                )

        output_lines.append(
            line
        )

    return "\n".join(
        output_lines
    )


# ============================================================
# Inspection / QC utilities
# ============================================================

def extract_uncertain_lines(text):
    if text is None:
        return []

    out = []

    for line in str(text).splitlines():
        if UNCERTAIN_RE.search(line):
            out.append({
                "line":
                    line,

                "uncertain_types":
                    find_uncertain_types(
                        line
                    ),

                "negative_types":
                    find_negative_types(
                        line
                    ),

                "would_mark":
                    not bool(
                        NEGATIVE_RE.search(
                            line
                        )
                    ),
            })

    return out


def summarize_uncertainty_texts(texts):
    total = 0
    containing_uncertainty = 0
    containing_negative = 0
    containing_both = 0

    type_counter = Counter()
    occurrence_counter = Counter()

    for text in texts:
        total += 1

        text = (
            ""
            if text is None
            else str(text)
        )

        has_u = (
            contains_uncertain_expression(
                text
            )
        )

        has_n = (
            contains_negative_expression(
                text
            )
        )

        containing_uncertainty += int(
            has_u
        )

        containing_negative += int(
            has_n
        )

        containing_both += int(
            has_u and has_n
        )

        for t in find_uncertain_types(
            text
        ):
            type_counter[t] += 1

        for k, v in (
            count_uncertain_type_occurrences(
                text
            ).items()
        ):
            occurrence_counter[k] += v

    return {
        "total_texts":
            total,

        "texts_with_uncertainty":
            containing_uncertainty,

        "texts_with_uncertainty_ratio":
            (
                containing_uncertainty
                / total
                if total
                else 0.0
            ),

        "texts_with_negative":
            containing_negative,

        "texts_with_both_uncertain_and_negative":
            containing_both,

        "case_count_by_uncertainty_type":
            dict(
                sorted(
                    type_counter.items()
                )
            ),

        "occurrence_count_by_uncertainty_type":
            dict(
                sorted(
                    occurrence_counter.items()
                )
            ),
    }
