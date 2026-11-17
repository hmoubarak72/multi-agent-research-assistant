import logging
import re
import structlog

SECRET_PATTERNS = [
    re.compile(r"sk-ant-[a-zA-Z0-9\-_]+"),
    re.compile(r"tvly-[a-zA-Z0-9\-_]+"),
    re.compile(r"Bearer\s+[a-zA-Z0-9\-_\.]+", re.IGNORECASE),
]


def _redact_secrets(_, __, event_dict):
    for key, value in list(event_dict.items()):
        if isinstance(value, str):
            sanitized = value
            for pattern in SECRET_PATTERNS:
                sanitized = pattern.sub("[REDACTED]", sanitized)
            event_dict[key] = sanitized
    return event_dict


def configure_logging() -> None:
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            _redact_secrets,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str):
    return structlog.get_logger(name)
