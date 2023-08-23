import logging
from logging import Logger
from rich.logging import RichHandler
from aibox.utils import as_path


def set_basic_config_rich(format="$(message)s", datefmt="[%X]"):
    logging.basicConfig(
        level=logging.NOTSET,
        format=format,
        datefmt=datefmt,
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )


def get_logger(
    name,
    add_file_handler=False,
    file_level=logging.ERROR,
    log_dir="event_logs",
) -> Logger:
    logger = logging.getLogger(name)

    if add_file_handler:
        log_path = as_path(log_dir) / name
        file_handler = logging.FileHandler(log_path.with_suffix(".log").as_posix())
        file_handler.setLevel(file_level)
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
