"""
Validation layer for the Query Rewriter.

The validator protects the retrieval pipeline from malformed,
hallucinated, or low-quality rewritten queries.

Design Principles
-----------------
- Single Responsibility Principle
- Open / Closed Principle
- Strategy Pattern
- Easy to extend
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel


# ==========================================================
# Validation Models
# ==========================================================


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ValidationResult(BaseModel):
    """
    Result returned by a validation rule.
    """

    passed: bool
    severity: Severity
    reason: str | None = None


# ==========================================================
# Base Rule
# ==========================================================


class ValidationRule(ABC):
    """
    Base interface for all validation rules.
    """

    @abstractmethod
    def validate(
        self,
        original_query: str,
        rewritten_query: str,
    ) -> ValidationResult:
        pass


# ==========================================================
# Rules
# ==========================================================


class EmptyRule(ValidationRule):

    def validate(self, original_query, rewritten_query):

        if not rewritten_query.strip():
            return ValidationResult(
                passed=False,
                severity=Severity.ERROR,
                reason="Rewrite is empty.",
            )

        return ValidationResult(
            passed=True,
            severity=Severity.INFO,
        )


class LengthRule(ValidationRule):

    def __init__(self, max_chars: int = 500):
        self.max_chars = max_chars

    def validate(self, original_query, rewritten_query):

        if len(rewritten_query) > self.max_chars:

            return ValidationResult(
                passed=False,
                severity=Severity.WARNING,
                reason=f"Rewrite exceeds {self.max_chars} characters.",
            )

        return ValidationResult(
            passed=True,
            severity=Severity.INFO,
        )


class PromptLeakageRule(ValidationRule):

    BANNED_PATTERNS = (
        "User Query:",
        "Rewritten Query:",
        "Answer:",
        "Explanation:",
        "Output:",
        "Original Query:",
    )

    def validate(self, original_query, rewritten_query):

        lower = rewritten_query.lower()

        for pattern in self.BANNED_PATTERNS:

            if pattern.lower() in lower:

                return ValidationResult(
                    passed=False,
                    severity=Severity.ERROR,
                    reason=f"Prompt leakage detected ({pattern})",
                )

        return ValidationResult(
            passed=True,
            severity=Severity.INFO,
        )


class SameQueryRule(ValidationRule):
    """
    Detects when the model simply echoes the original query.
    """

    def validate(self, original_query, rewritten_query):

        if (
            original_query.strip().lower()
            == rewritten_query.strip().lower()
        ):

            return ValidationResult(
                passed=False,
                severity=Severity.WARNING,
                reason="Rewrite identical to original query.",
            )

        return ValidationResult(
            passed=True,
            severity=Severity.INFO,
        )


# ==========================================================
# Validator
# ==========================================================


class RewriteValidator:
    """
    Executes validation rules sequentially.

    Stops immediately on ERROR.

    WARNINGS are collected but do not stop execution.
    """

    def __init__(
        self,
        rules: list[ValidationRule],
    ):
        self.rules = rules

    def validate(
        self,
        original_query: str,
        rewritten_query: str,
    ) -> ValidationResult:

        warnings = []

        for rule in self.rules:

            result = rule.validate(
                original_query,
                rewritten_query,
            )

            if (
                not result.passed
                and result.severity == Severity.ERROR
            ):
                return result

            if (
                not result.passed
                and result.severity == Severity.WARNING
            ):
                warnings.append(result.reason)

        if warnings:

            return ValidationResult(
                passed=True,
                severity=Severity.WARNING,
                reason="\n".join(warnings),
            )

        return ValidationResult(
            passed=True,
            severity=Severity.INFO,
        )


# ==========================================================
# Factory
# ==========================================================


def build_default_validator() -> RewriteValidator:
    """
    Returns the default validator used in production.
    """

    return RewriteValidator(
        rules=[
            EmptyRule(),
            PromptLeakageRule(),
            LengthRule(max_chars=500),
            SameQueryRule(),
        ]
    )