"""
This module contains functions for validating rewritten queries. It checks for various conditions such as length,
banned patterns, and whether the rewrite is empty.
The main function `validate_rewrite` returns a `RewriteResult` object that indicates whether the rewrite is valid or if a fallback should be used.
"""
from .models import RewriteResult

#___constants ___________________________________
MAX_QUERY_WORDS = 40
BANNED_PATTERNS = (
    "User Query:",
    "Rewritten Query:",
    "Output:",
    "Example:",
    "Answer:",
    "```",
    "###",
)

#___functions ___________________________________
def validate_rewrite(
        original: str,
        rewritten: str,
) -> RewriteResult:
    rewrite = rewrite.strip()

    if not rewrite:
        return RewriteResult(
            original_query=original,
            retrieval_query=original,
            used_fallback=True,
            fallback_reason="empty",
            original_length=len(original),
            rewritten_length=0,
        )

    if any(x.lower() in rewrite.lower() for x in BANNED_PATTERNS):
        return RewriteResult(
            original_query=original,
            retrieval_query=original,
            used_fallback=True,
            fallback_reason="pattern_detected",
            original_length=len(original),
            rewritten_length=len(rewrite),
        )

    if len(rewrite.split()) > MAX_QUERY_WORDS:
        return RewriteResult(
            original_query=original,
            retrieval_query=original,
            used_fallback=True,
            fallback_reason="too_long",
            original_length=len(original),
            rewritten_length=len(rewrite),
        )

    if len(rewrite) > len(original) * 3:
        return RewriteResult(
            original_query=original,
            retrieval_query=original,
            used_fallback=True,
            fallback_reason="expanded_too_much",
            original_length=len(original),
            rewritten_length=len(rewrite),
        )

    return RewriteResult(
        original_query=original,
        retrieval_query=rewrite,
        original_length=len(original),
        rewritten_length=len(rewrite),
    )
