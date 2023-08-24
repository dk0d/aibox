from typing import Iterable, Callable, Sequence
from rich.style import StyleType
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    ProgressType,
    TaskProgressColumn,
    TimeElapsedColumn,
)


def track(
    sequence: Sequence[ProgressType] | Iterable[ProgressType],
    description: str = "Working...",
    total: float | None = None,
    auto_refresh: bool = True,
    console: Console | None = None,
    transient: bool = False,
    get_time: Callable[[], float] | None = None,
    refresh_per_second: float = 10,
    style: StyleType = "bar.back",
    complete_style: StyleType = "bar.complete",
    finished_style: StyleType = "bar.finished",
    pulse_style: StyleType = "bar.pulse",
    update_period: float = 0.1,
    disable: bool = False,
    show_speed: bool = True,
) -> Iterable[ProgressType]:
    """Track progress by iterating over a sequence.

    Args:
        sequence (Iterable[ProgressType]): A sequence (must support "len") you wish to iterate over.
        description (str, optional): Description of task show next to progress bar. Defaults to "Working".
        total: (float, optional): Total number of steps. Default is len(sequence).
        auto_refresh (bool, optional): Automatic refresh, disable to force a refresh after each iteration. Default is True.
        transient: (bool, optional): Clear the progress on exit. Defaults to False.
        console (Console, optional): Console to write to. Default creates internal Console instance.
        refresh_per_second (float): Number of times per second to refresh the progress information. Defaults to 10.
        style (StyleType, optional): Style for the bar background. Defaults to "bar.back".
        complete_style (StyleType, optional): Style for the completed bar. Defaults to "bar.complete".
        finished_style (StyleType, optional): Style for a finished bar. Defaults to "bar.finished".
        pulse_style (StyleType, optional): Style for pulsing bars. Defaults to "bar.pulse".
        update_period (float, optional): Minimum time (in seconds) between calls to update(). Defaults to 0.1.
        disable (bool, optional): Disable display of progress.
        show_speed (bool, optional): Show speed if total isn't known. Defaults to True.
    Returns:
        Iterable[ProgressType]: An iterable of the values in the sequence.

    """

    # columns: List["ProgressColumn"] = [TextColumn("[progress.description]{task.description}")] if description else []
    # columns.extend(
    #     (
    #         BarColumn(
    #             style=style,
    #             complete_style=complete_style,
    #             finished_style=finished_style,
    #             pulse_style=pulse_style,
    #         ),
    #         TaskProgressColumn(show_speed=show_speed),
    #         TimeRemainingColumn(elapsed_when_finished=True),
    #     )
    # )
    columns = [
        TextColumn("[bold blue][progress.description]{task.description}", justify="right"),
        "•",
        BarColumn(
            style=style,
            complete_style=complete_style,
            finished_style=finished_style,
            pulse_style=pulse_style,
        ),
        "•",
        TaskProgressColumn(show_speed=show_speed),
        "•",
        TextColumn("{task.completed:.0f}/{task.total:.0f}", justify="right"),
        "•",
        TimeRemainingColumn(),
        "(",
        TimeElapsedColumn(),
        ")",
    ]
    progress = Progress(
        *columns,
        auto_refresh=auto_refresh,
        console=console,
        transient=transient,
        get_time=get_time,
        refresh_per_second=refresh_per_second or 10,
        disable=disable,
    )

    with progress:
        yield from progress.track(sequence, total=total, description=description, update_period=update_period)
