"""Exception hierarchy with stable `kind` values."""

from __future__ import annotations


class SchedulerError(Exception):
    """Base class for every error this package raises on purpose."""

    kind = "scheduler_error"

    def __init__(self, message: str, **context: object) -> None:
        super().__init__(message)
        self.message = message
        self.context = {key: value for key, value in context.items() if value is not None}

    def to_document(self) -> dict[str, object]:
        document: dict[str, object] = {"error": self.kind, "message": self.message}
        document.update(self.context)
        return document


class ParseError(SchedulerError):
    """Malformed input documents."""

    kind = "parse_error"

    def __init__(self, message: str, *, line: int | None = None, **context: object) -> None:
        super().__init__(message, line=line, **context)


class ValidationError(SchedulerError):
    """A request that is well-formed but not allowed: unknown node, bad policy, duplicate id."""

    kind = "validation_error"


class SchedulingError(SchedulerError):
    """A schedule that cannot be produced for the given input."""

    kind = "scheduling_error"


class OutputError(SchedulerError):
    """The output target cannot be used safely."""

    kind = "output_error"
