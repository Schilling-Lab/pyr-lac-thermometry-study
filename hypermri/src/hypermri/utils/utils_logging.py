from dataclasses import dataclass


@dataclass
class log_modes_generator:
    critical: int = 50  # logging.CRITICAL
    error: int = 40  # logging.ERROR
    warning: int = 30  # logging.WARNING
    info: int = 20  # logging.INFO
    debug: int = 10  # logging.DEBUG
    notset: int = 0  # logging.NOTSET

    def __getitem__(cls, key):
        if isinstance(key, int):
            return key
        if isinstance(key, str):
            return getattr(cls, key.lower())
        raise ValueError(f"'{key}' is not a valid logging mode.")


LOG_MODES = log_modes_generator()


def init_default_logger(
    name, default_mode="error", fstring="[%(levelname)s] %(name)s: %(message)s "
):
    """Initialize the default logger class.

    fstring:
        Format of the log message, example: '%(asctime)s:%(name)s:%(message)s '
    """
    import logging
    logger = logging.getLogger(name)

    logger.setLevel(LOG_MODES[default_mode])

    formatter = logging.Formatter(fstring)
    # formatter = logging.Formatter()

    # Multiple handlers (to stream and/or to file) are possible.
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # we have to ensure we don't add a handler every time the class is called.
    if not logger.hasHandlers():
        logger.addHandler(stream_handler)

    return logger