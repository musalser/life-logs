class HTRError(Exception):
    """Base error for the handwriting recognition/training module."""


class NotFoundError(HTRError):
    pass


class PageStateError(HTRError):
    """Operation is not allowed for the current page status."""


class InvalidTranscriptionError(HTRError):
    """Transcription violates validation rules (e.g. implicit empty line)."""


class CorruptImageError(HTRError):
    """Image cannot be opened/decoded."""


class DatasetBuildError(HTRError):
    """Training dataset could not be built."""


class TrainingError(HTRError):
    """Fine-tuning failed inside the HTR backend."""
