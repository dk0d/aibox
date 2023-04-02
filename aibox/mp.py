import concurrent.futures
import itertools
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Callable, Optional, List

import tqdm


def multiprocess(
    func,
    kwargsList: List[dict],
    poolMode="process",  # or 'thread'
    onResult: Optional[Callable] = None,
    maxJobs=400,
    desc="Processing",
    showProg=True,
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
        maxJobs (int): max number of jobs to queue, adjust this if you get out of memory errors, especially for process pools on CPU
        desc (str): Description to display on progress bar 
        showProg (bool): flag whether to show progess bar

    """
    if showProg:
        prog = tqdm.tqdm(total=len(kwargsList), desc=desc)
    else:
        prog = None
        
    if poolMode == "process":
        ProcessPool = ProcessPoolExecutor
    else:
        ProcessPool = ThreadPoolExecutor

    iterArgs = iter(kwargsList)
    
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
