import re
from pathlib import Path
from pprint import pprint
from sys import version
from typing import Callable, List, Optional

import pandas as pd
import tqdm

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import tensorboard as tb

    class TBLogReader:
        def __init__(self, log_root_dir: Path = Path("logs")):
            major_ver, minor_ver, _ = version.parse(tb.__version__).release
            assert (
                major_ver >= 2 and minor_ver >= 3
            ), "This class requires TensorBoard 2.3 or later"

            self.log_root_dir = log_root_dir
            self.allEventFiles = self._getLogsInPath(self.log_root_dir)
            # self.dataFrames = self._events2DataFrames(self.allEventFiles)

        def getLogs(self, path_regex_mask: Optional[str] = None) -> pd.DataFrame:
            if path_regex_mask is not None:
                regex = re.compile(path_regex_mask)
                event_files = [
                    p
                    for p in self.allEventFiles
                    if regex.search(
                        p.relative_to(self.log_root_dir).parts[0].lower(),
                    )
                    is not None
                ]
            else:
                event_files = self.allEventFiles
            return self._events2DataFrames(event_files)

        @staticmethod
        def _getLogsInPath(path: Path):
            return list(path.rglob("event*"))

        # Extraction function
        @staticmethod
        def _eventPath2DataFrame(path: Path) -> Optional[pd.DataFrame]:
            """convert single tensorflow log file to pandas DataFrame

            Parameters
            ----------
            path : str
                path to tensorflow log file

            Returns
            -------
            pd.DataFrame
                converted dataframe
            """
            DEFAULT_SIZE_GUIDANCE = {
                "compressedHistograms": 1,
                "images": 1,
                "scalars": 0,  # 0 means load all
                "histograms": 1,
            }
            # runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
            try:
                event_acc = EventAccumulator(path.as_posix(), DEFAULT_SIZE_GUIDANCE)
                event_acc.Reload()
                tags = event_acc.Tags()["scalars"]
                tags.sort()
                runlog_data = pd.DataFrame()
                for tag in tags:
                    if tag == "hp_metric":
                        continue
                    event_list = event_acc.Scalars(tag)
                    events = list(
                        map(lambda x: (x.step, x.value, x.wall_time), event_list)
                    )
                    events = pd.DataFrame(
                        events, columns=[[tag] * 3, ["step", "value", "wall_time"]]
                    )
                    events.set_index((tag, "step"), inplace=True)
                    runlog_data = pd.concat((runlog_data, events), axis=1)
                columns = set(runlog_data.columns)
                columns.remove(("epoch", "wall_time"))
                runlog_data.drop_duplicates(subset=columns, inplace=True)
                df = pd.DataFrame()
                for col in runlog_data.columns.to_flat_index():
                    if col == ("epoch", "wall_time"):
                        df["wall_time"] = runlog_data[col]
                    elif "value" in col[-1]:
                        df[col[0]] = runlog_data[col]
                return df
            # Dirty catch of DataLossError
            except Exception as e:
                print(f"Event file possibly corrupt: {path}")
                # print(e)
                # traceback.print_exc()
            return None

        @staticmethod
        def _events2DataFrames(event_paths: List[Path]) -> pd.DataFrame:
            all_logs = {}
            event_paths.sort()
            for path in tqdm.tqdm(event_paths, desc="Loading Events"):
                name = path.parent.parent.name
                if name not in all_logs.keys():
                    all_logs[name] = {}

                fold = int(path.parent.name.split("_")[-1])

                if fold not in all_logs[name].keys():
                    all_logs[name][fold] = []

                df = TBLogHelper._eventPath2DataFrame(path)

                if df is not None:
                    df["model"] = name
                    df["fold"] = fold
                    df["eventIdx"] = len(all_logs[name][fold])
                    all_logs[name][fold].append(df)
                else:
                    all_logs[name][fold].append(None)

            return pd.concat(
                [
                    pd.concat(dfs)
                    for m, fold in all_logs.items()
                    for fIdx, dfs in fold.items()
                ]
            ).reset_index(drop=True)

except ImportError:
    pass
