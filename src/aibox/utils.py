from dataclasses import dataclass
from pathlib import Path
from pprint import pformat

import yaml
from rich import print as rprint

from .config import class_from_string, config_from_path



try:
    import mlflow

    def search_runs(
        client: mlflow.MlflowClient,
        run_name=None,
        experiment_ids: list[str] | None = None,
        max_results=1,
        filter_string: str | None = None,
    ):
        if experiment_ids is None:
            experiment_ids = [e.experiment_id for e in client.search_experiments()]

        filters = [(f"attributes.run_name LIKE '%{run_name}%'" if run_name is not None else None)]

        if filter_string is not None:
            filters.append(filter_string)

        filters = [f for f in filters if f is not None]

        filter_string = " AND ".join(f for f in filters if f is not None) if len(filters) > 0 else ""

        runs = client.search_runs(
            experiment_ids,
            filter_string=filter_string,
            max_results=max_results,
            order_by=["metrics.`test/loss` DESC"],
        )

        if max_results == 1 and len(runs) == 1:
            return runs[0]

        return runs

    def load_model_from_run(run, tracking_uri, config, alias="best"):
        try:
            model_dir = mlflow.artifacts.download_artifacts(
                run_id=run.info.run_id,
                artifact_path="model",
                tracking_uri=tracking_uri,
            )
            checkpoints = get_mlflow_checkpoints(Path(model_dir))
            best = [c.path for c in checkpoints if alias in c.aliases][-1]
            Model = class_from_string(config.model.class_path)
            model = Model.load_from_checkpoint(best, **config.model.args)
            return model
        except Exception as e:
            print(f"Error loading model for run ID: {run.info.run_id}")
            print(e)

    def load_config_from_run(run, tracking_uri, config_file="config.yml"):
        try:
            config_path = mlflow.artifacts.download_artifacts(
                run_id=run.info.run_id,
                artifact_path=config_file,
                tracking_uri=tracking_uri,
            )
            config = config_from_path(config_path)
            return config
        except Exception as e:
            print(f"Error loading config for run ID: {run.info.run_id}")
            print(e)

except ImportError:
    pass


@dataclass
class MLFlowCheckpointEntry:
    path: Path
    metadata: dict
    aliases: list[str]

    def _get_metadata(self, k: str):
        return self.metadata["Checkpoint"]["k"]

    @property
    def mode(self) -> list[Path]:
        return self._get_metadata("mode")

    @property
    def monitor(self):
        return self._get_metadata("monitor")

    @property
    def save_last(self):
        return self._get_metadata("save_last")

    @property
    def save_top_k(self):
        return self._get_metadata("save_top_k")

    @property
    def save_weights_only(self):
        return self._get_metadata("save_weights_only")

    @property
    def original_filename(self):
        return self.metadata["original_filename"]

    @property
    def score(self):
        return self.metadata["score"]

    @property
    def name(self):
        return self.path.parent.parent.name

    @property
    def run_dir(self) -> Path:
        # assumes self.path is of the form: mlruns/<exp_id>/<run_id>/artifacts/model/checkpoints/<checkpoint_id>
        return self.path.parent.parent.parent.parent.parent

    @property
    def run_meta_path(self) -> Path:
        # assumes self.path is of the form: mlruns/<exp_id>/<run_id>/artifacts/model/checkpoints/<checkpoint_id>
        return self.run_dir / "meta.yaml"

    @property
    def run_id(self):
        # assumes path is of the form: mlruns/<exp_id>/<run_id>/artifacts/model/checkpoints/<checkpoint_id>
        return self.run_dir.name

    @property
    def exp_id(self):
        return self.run_dir.parent.name


def rewrite_artifact_path(metadata_file, pwd, artifact_path_key):
    with open(metadata_file, "r") as f:
        y = yaml.safe_load(f)
        y[artifact_path_key] = f"file://{pwd}"

    with open(metadata_file, "w") as f:
        print(yaml.dump(y, default_flow_style=False, sort_keys=False))
        yaml.dump(y, f, default_flow_style=False, sort_keys=False)


def fix_mlflow_artifact_paths(mlflow_root: Path):
    for experiment_folder in mlflow_root.expanduser().resolve().iterdir():
        metadata_file = experiment_folder / "meta.yaml"

        # Fix experiment metadata
        if metadata_file.exists():
            rewrite_artifact_path(metadata_file, experiment_folder, artifact_path_key="artifact_location")
        for run_folder in experiment_folder.iterdir():
            metadata_file = run_folder / "meta.yaml"
            print(run_folder)

            # Fix run metadata
            if metadata_file.exists():
                rewrite_artifact_path(metadata_file, run_folder / "artifacts", artifact_path_key="artifact_uri")


def get_mlflow_run_metas(mlruns_root: Path):
    _configs = [
        run_meta
        for exp_dir in mlruns_root.glob("*")
        if exp_dir.is_dir()
        for run_dir in exp_dir.glob("*")
        if run_dir.is_dir()
        for run_meta in run_dir.glob("*.yaml")
    ]
    _configs = [yaml.load(c.read_text(), Loader=yaml.SafeLoader) for c in _configs]
    _configs = [c for c in _configs if "run_id" in c]
    return _configs


def get_mlflow_checkpoints(mlruns_root: Path):
    """
    Returns a list of MLFlowCheckpointEntry objects

    Assumes that mlruns_root is the root of the mlruns folder and that the checkpoints
    were saved using the CombinedLogger
    """
    mlruns_root = mlruns_root.resolve()
    checkpoints = [
        MLFlowCheckpointEntry(
            path=p,
            metadata=yaml.load((p.parent / "metadata.yaml").read_text(), Loader=yaml.SafeLoader),
            aliases=yaml.load((p.parent / "aliases.txt").read_text(), Loader=yaml.SafeLoader),
        )
        for p in mlruns_root.glob("**/*.ckpt")
    ]
    return checkpoints


# mlruns_root = Path("logs/mlruns")
#
# # print(get_run_metas(mlruns_root))
#
# client = mlflow.MlflowClient(mlruns_root.as_posix())
# experiments = client.search_experiments()
# # print(experiments)
# # print(runs[0].data)
# # print([client.list_artifacts(r.info.run_id) for r in runs])
# experiment = client.get_experiment_by_name("debug")
# assert experiment is not None
# print(experiment)
# runs = client.search_runs(experiment_ids=[experiment.experiment_id])
# run = runs[0]
# artifacts = client.list_artifacts(run.info.run_id)
# # paths = client.download_artifacts(run.info.run_id, "model")
#
# model_dir = mlflow.artifacts.download_artifacts(
#     run_id=run.info.run_id, artifact_path="model", tracking_uri=mlruns_root.as_posix()
# )
# print(model_dir)
# checkpoints = get_checkpoints(Path(model_dir))
# latest = [c for c in checkpoints if "best" in c.aliases][-1]
# print(latest)
# config_path = mlflow.artifacts.download_artifacts(
#     run_id=run.info.run_id, artifact_path="config.yml", tracking_uri=mlruns_root.as_posix()
# )
# print(config_path)


def print(*args, **kwargs):
    args = [a if isinstance(a, str) else pformat(a) for a in args]
    rprint(*args, **kwargs)
