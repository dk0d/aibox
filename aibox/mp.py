import concurrent.futures
import itertools
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Callable, Optional

import tqdm


def multiprocess(
    func,
    argsList,
    poolMode="process",  # or 'thread'
    onResult: Optional[Callable] = None,
    maxJobs=400,
    desc="Processing",
    showProg=True,
):
    """

    Args:
        func ():
        argsList ():
        poolMode ():
        onResult (): Callback when result collected, takes in result object and the progress bar for updating post-fix or other operations.
        maxJobs ():
        desc ():
        showProg ():

    Returns:

    """
    iterArgs = iter(argsList)
    if showProg:
        prog = tqdm.tqdm(total=len(argsList), desc=desc)
    else:
        prog = None
    if poolMode == "process":
        ProcessPool = ProcessPoolExecutor
    else:
        ProcessPool = ThreadPoolExecutor

    with ProcessPool() as pool:
        futures = {pool.submit(func, **a) for a in itertools.islice(iterArgs, maxJobs)}

        while futures:
            done, futures = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )
            # try:
            for f in done:
                res = f.result()
                if onResult is not None:
                    onResult(res, prog)
                if prog is not None:
                    prog.update()

            for a in itertools.islice(iterArgs, len(done)):
                futures.add(pool.submit(func, **a))
            # except KeyboardInterrupt:
            #     for f in done:
            #         res = f.result()
            #         if onResult is not None:
            #             onResult(res, prog)
            #         if prog is not None:
            #             prog.update()
