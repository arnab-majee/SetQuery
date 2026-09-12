class SatQueryError(Exception):
    """Base exception for SatQuery AI."""


class ModelServiceError(SatQueryError):
    """Raised when a model service fails."""


class UnsupportedTaskError(SatQueryError):
    """Raised when a requested task is not implemented yet."""


class InvalidInputError(SatQueryError):
    """Raised when input validation fails."""