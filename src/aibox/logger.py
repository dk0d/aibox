import datetime
import logging
from pathlib import Path
from logging import Logger

from rich.logging import RichHandler


class LogTic:
    def __init__(self, msg=None, logger=None):
        self.start = datetime.datetime.now()
        if logger is None:
            self.printer = print
        else:
            self.printer = logger.info

        if msg is not None:
            print(msg)

    def __call__(self, msg):
        out = f"{msg} ({str(datetime.datetime.now() - self.start)} elapsed)"
        self.printer(out)
        self.start = datetime.datetime.now()


def set_basic_config_rich(format="%(message)s", datefmt="[%X]"):
    logging.basicConfig(
        level=logging.INFO,
        format=format,
        datefmt=datefmt,
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )


def get_logger(
    name,
    add_file_handler=False,
    file_level=logging.NOTSET,
    log_dir="pylogs",
) -> Logger:
    set_basic_config_rich()
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if add_file_handler:
        log_dir = Path(log_dir).expanduser().resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / name
        file_handler = logging.FileHandler(log_path.with_suffix(".log").as_posix())
        file_handler.setLevel(file_level)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)-8s - %(message)s"))
        logger.addHandler(file_handler)

    return logger


# TODO: Setup default logging

# def load_log_config(path: Path):
#     import yaml
#     with open(path, "r") as f:
#         config = yaml.load(f, Loader=yaml.FullLoader)
#     logging.config.dictConfig(config)


# def get_logger(name):
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)
# return logger
