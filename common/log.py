import logging
import os
import sys

from tqdm import tqdm


class _TqdmHandler(logging.Handler):
    def emit(self, record):
        try:
            tqdm.write(self.format(record), file=sys.stdout)
        except Exception:
            self.handleError(record)


def get_logger(tag, log_file=None):
    logger = logging.getLogger(tag)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(name)s | %(message)s", "%H:%M:%S")
    handler = _TqdmHandler()
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    if log_file:
        os.makedirs(os.path.dirname(str(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


def progress(total, desc, initial=0, unit="it", leave=True):
    interval = float(os.environ.get("PA3_TQDM_INTERVAL", "1"))
    return tqdm(
        total=total,
        desc=desc,
        initial=initial,
        unit=unit,
        file=sys.stdout,
        mininterval=interval,
        maxinterval=max(interval, 10.0),
        dynamic_ncols=False,
        ncols=int(os.environ.get("PA3_TQDM_COLS", "180")),
        leave=leave,
        smoothing=0.05,
    )


def fmt_metrics(metrics):
    out = {}
    for key, value in metrics.items():
        out[key] = f"{value:.4g}" if isinstance(value, float) else str(value)
    return out


def line(metrics):
    return " | ".join(f"{k}={v}" for k, v in fmt_metrics(metrics).items())
