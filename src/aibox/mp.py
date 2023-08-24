# %%
import concurrent.futures
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Callable, Any
import time

from pandas.io.formats.printing import justify

from aibox.logger import get_logger
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
)


def _initializer_mute():
    sys.stdout = open(os.devnull, "w")


LOGGER = get_logger(__name__)


def multiprocess(
    func,
    kwargsList: list[dict],
    poolMode="process",  # or 'thread'
    onResult: Callable[[Any, Progress | None], None] | None = None,
    maxJobs=400,
    maxWorkers=None,
    desc="Processing",
    mute=True,
    showProg=True,
    stopAllOnException=False,
):
    """
    Multi-processing helper for handling job submission and
    limiting memory consumption by queuing at max the specified number of jobs

    Args:
        func (Callable): function to be called
        kwargsList (List[dict]): list of keyword argument dictionaries that will be past into the function
        poolMode (str): one of 'process' or 'thread'
        onResult (Optional[Callable]): Callback when result collected, takes in result object and the progress bar
            for updating post-fix or other operations.
        maxJobs (int): max number of jobs to queue, adjust this if you get out of memory errors, especially for process
            pools on CPU
        desc (str): Description to display on progress bar
        showProg (bool): flag whether to show progess bar

    """
    # prog = track(total=len(kwargsList), description=desc)

    progress = Progress(
        TextColumn("[bold blue][progress.description]{task.description}", justify="right"),
        "•",
        BarColumn(bar_width=None),
        "•",
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        TextColumn("{task.completed:.0f}/{task.total:.0f}", justify="right"),
        "•",
        TimeRemainingColumn(),
        "(",
        TimeElapsedColumn(),
        ")",
        refresh_per_second=1,
        # WARN: the console doesn't get detected sometimes here, so it's forced
        # Needs to be checked for issues
        console=Console(file=sys.stdout, force_terminal=True),
    )

    if poolMode == "process":
        ProcessPool = ProcessPoolExecutor
    else:
        ProcessPool = ThreadPoolExecutor

    poolKwds: dict = {
        "max_workers": maxWorkers,
    }

    if mute:
        poolKwds["initializer"] = _initializer_mute

    iterArgs = iter(kwargsList)

    with progress:
        task = progress.add_task(desc, start=True, total=len(kwargsList), visible=showProg)

        with ProcessPool(**poolKwds) as pool:
            # with ProcessPool(**poolKwds) as pool:
            futures = {pool.submit(func, **a) for a in itertools.islice(iterArgs, maxJobs)}
            done = []  # to help with graceful exit
            try:
                while futures:
                    done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                    for f in done:
                        try:
                            res = f.result()
                            if onResult is not None:
                                onResult(res, progress)
                        except Exception as e:
                            LOGGER.exception("Exception detected, cancelling all jobs...")
                            if stopAllOnException:
                                raise e
                            try:
                                if not f.cancelled():
                                    f.cancel()
                            except Exception:
                                pass

                        progress.update(task, advance=1.0, refresh=True)

                    for a in itertools.islice(iterArgs, len(done)):
                        futures.add(pool.submit(func, **a))

            except KeyboardInterrupt:
                LOGGER.exception("\nKeyboard interrupt detected, cancelling jobs...")

                canceling = progress.add_task("Cancelling jobs", total=len(futures))
                collecting = progress.add_task("Collecting results", total=len(done))

                for f in futures:
                    f.cancel()
                    progress.update(canceling, advance=1.0, refresh=True)

                for f in done:
                    res = f.result()
                    if onResult is not None:
                        onResult(res, progress)

                    progress.update(collecting, advance=1.0, refresh=True)
