import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tqdm
from google.protobuf import text_format

# from pprint import pprint
from packaging import version
from tensorboard.compat import tf

try:
    import tensorboard as tb
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tensorboard.plugins.projector.projector_plugin import ProjectorConfig

    class TBLogReader:
        def __init__(self, log_root_dir: Path = Path("logs")):
            major_ver, minor_ver, _ = version.parse(tb.__version__).release
            assert major_ver >= 2 and minor_ver >= 3, "This class requires TensorBoard 2.3 or later"

            self.log_root_dir = log_root_dir.expanduser().resolve()
            assert self.log_root_dir.exists(), f"Log directory not found: {self.log_root_dir}"
            self.allEventFiles = self._get_logs_in_path(self.log_root_dir)
            # self.dataFrames = self._events2DataFrames(self.allEventFiles)

        @property
        def versions(self):
            parents = list(set(p.parent.name for p in self.allEventFiles))
            parents.sort()
            return parents

        def _get_sprite_image(run, name):
            pass

        def _read_metadata_tsv_file(self, path, num_rows: Optional[int] = None):
            if not Path(path).is_relative_to(self.log_root_dir):
                path = self.log_root_dir / path

            num_header_rows = 0
            with tf.io.gfile.GFile(f"{path}", "r") as f:
                lines = []
                # Stream reading the file with early break
                # in case the file doesn't fit in memory.
                for line in f:
                    lines.append(line)
                    if len(lines) == 1 and "\t" in lines[0]:
                        num_header_rows = 1
                    if num_rows is not None and len(lines) >= num_rows + num_header_rows:
                        break

            return "".join(lines)

        def _read_tensor_tsv_file(self, path):
            """Ported from tensorboard ProjectorPlugin._read_tensor_tsv_file(fpath)

            Args:
                path (_type_): _description_
            """
            if not Path(path).is_relative_to(self.log_root_dir):
                path = self.log_root_dir / path

            with tf.io.gfile.GFile(f"{path}", "r") as f:
                tensor = []
                for line in f:
                    line = line.rstrip("\n")
                    if line:
                        tensor.append(list(map(float, line.split("\t"))))
            return np.array(tensor, dtype="float32")

        def get_embeddings(self, version: Optional[str] = None):
            if version is None:
                projector_configs = list(self.log_root_dir.rglob("*.pbtxt"))
            else:
                v_path = self.log_root_dir / version
                if not v_path.exists():
                    print(f"{v_path} not found")
                    return []
                projector_configs = v_path.rglob("*.pbtxt")
            embeddings = {}
            for config_path in projector_configs:
                config = ProjectorConfig()
                with tf.io.gfile.GFile(config_path.as_posix(), "r") as f:
                    file_content = f.read()
                text_format.Parse(file_content, config)
                embeddings[config_path.parent.name] = [
                    (
                        e.tensor_name,
                        self._read_tensor_tsv_file(config_path.parent / e.tensor_path),
                        self._read_metadata_tsv_file(config_path.parent / e.metadata_path),
                    )
                    for e in config.embeddings
                ]
            return embeddings

        def get_logs(self, path_regex_mask: Optional[str] = None) -> Optional[pd.DataFrame]:
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
            if len(event_files) == 0:
                return None
            return self._events_to_dfs(event_files)

        @staticmethod
        def _get_logs_in_path(path: Path):
            return list(path.rglob("event*"))

        # Extraction function
        @staticmethod
        def _event_path_to_df(path: Path) -> Optional[pd.DataFrame]:
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
                    events = list(map(lambda x: (x.step, x.value, x.wall_time), event_list))
                    events = pd.DataFrame(events, columns=[[tag] * 3, ["step", "value", "wall_time"]])
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
            except Exception:
                print(f"Event file possibly corrupt: {path}")
                # print(e)
                # traceback.print_exc()
            return None

        @staticmethod
        def _events_to_dfs(event_paths: list[Path]) -> pd.DataFrame:
            all_logs = {}
            event_paths.sort()
            for path in tqdm.tqdm(event_paths, desc="Loading Events"):
                name = path.parent.parent.name
                if name not in all_logs.keys():
                    all_logs[name] = {}

                fold = int(path.parent.name.split("_")[-1])

                if fold not in all_logs[name].keys():
                    all_logs[name][fold] = []

                df = TBLogReader._event_path_to_df(path)

                if df is not None:
                    df["model"] = name
                    df["fold"] = fold
                    df["eventIdx"] = len(all_logs[name][fold])
                    all_logs[name][fold].append(df)
                else:
                    all_logs[name][fold].append(None)

            return pd.concat(
                [pd.concat(dfs) for m, fold in all_logs.items() for fIdx, dfs in fold.items()]
            ).reset_index(drop=True)

except ImportError:
    pass
