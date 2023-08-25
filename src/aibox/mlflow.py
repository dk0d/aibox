from dataclasses import dataclass
from pathlib import Path

import lightning as L
import mlflow
import yaml
from mlflow.entities import Run
from mlflow.store.entities.paged_list import PagedList

from .config import (
    Config,
    class_from_string,
    config_from_dotlist,
    config_from_path,
    derive_args,
    derive_classpath,
)
from .torch.utils import get_device


def search_runs(
    tracking_uri: str,
    registry_uri: str | None = None,
    *,
    run_name=None,
    experiment_ids: list[str] | None = None,
    experiment_name: str | None = None,
    max_results=1,
    filter_string: str | None = None,
    order_by: list[str] | None = None,
    page_token: str | None = None,  # get from paged list
) -> PagedList[Run]:
    """Search runs in MLFlow
    for more on search params see: https://mlflow.org/docs/latest/search-runs.html

    Args:
        tracking_uri (str): [description]
        registry_uri (str, optional): [description]. Defaults to None.
        run_name ([type], optional): [description]. Defaults to None.
        experiment_ids (list[str], optional): [description]. Defaults to None.
        max_results (int, optional): [description]. Defaults to 1.
        filter_string (str, optional): [description]. Defaults to None.
        order_by (list[str], optional): [description]. Defaults to None.
        page_token (str, optional): get from PagedList. Defaults to None.
    """
    client: mlflow.MlflowClient = mlflow.MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)

    if experiment_ids is None:
        exp_filter_string = None if experiment_name is None else f"attributes.name LIKE '%{experiment_name}%'"
        experiment_ids = [e.experiment_id for e in client.search_experiments(filter_string=exp_filter_string)]

    filters = [(f"attributes.run_name LIKE '%{run_name}%'" if run_name is not None else None)]

    if filter_string is not None:
        filters.append(filter_string)

    filters = [f for f in filters if f is not None]

    filter_string = " AND ".join(f for f in filters if f is not None) if len(filters) > 0 else ""

    order_by = order_by if order_by is not None else ["metrics.`test/loss` DESC"]

    runs = client.search_runs(
        experiment_ids,
        filter_string=filter_string,
        max_results=max_results,
        order_by=["metrics.`test/loss` DESC"],
        page_token=page_token,
    )

    return runs


def get_runs(
    tracking_uri: str,
    registry_uri: str | None = None,
    *,
    run_name=None,
    experiment_ids: list[str] | None = None,
    max_results=1000,
    filter_string: str | None = None,
    order_by: list[str] | None = None,
    page_token: str | None = None,  # get from paged list
) -> PagedList[Run]:
    runs = search_runs(
        tracking_uri,
        registry_uri,
        run_name=run_name,
        experiment_ids=experiment_ids,
        max_results=max_results,
        filter_string=filter_string,
        order_by=order_by,
        page_token=page_token,
    )
    return runs


def load_ckpt_from_run(
    run: Run | str,
    tracking_uri: str,
    alias="best",
    new_root=None,
    **model_kwargs,
) -> tuple[Config, L.LightningModule]:
    if isinstance(run, str):  # is run_id
        client = mlflow.MlflowClient(tracking_uri=tracking_uri)
        run = client.get_run(run_id=run)

    model_dir = mlflow.artifacts.download_artifacts(
        run_id=run.info.run_id,
        artifact_path="model",
        tracking_uri=tracking_uri,
    )
    config = load_config_from_run(run, tracking_uri, new_root=new_root)
    checkpoints = get_mlflow_checkpoints(Path(model_dir))
    best = [c.path for c in checkpoints if alias in c.aliases][-1]
    # try:
    if not hasattr(config, "model"):
        raise Exception("Config has no model key")
    model_cfg = getattr(config, "model")
    Model: L.LightningModule = class_from_string(derive_classpath(model_cfg))
    device = get_device()
    model_kwargs.update(map_location=device)
    model = Model.load_from_checkpoint(best, **derive_args(model_cfg, **model_kwargs))
    # except Exception as e:
    # raise Exception(f"Unable to load model for run: {run.info.run_id}\n{e}")
    return config, model


def load_config_from_run(run: Run, tracking_uri: str, config_file="config.yml", new_root=None):
    from .utils import resolve_paths

    # try:
    # param_config = config_from_dotlist([f"{k}={v}" for k, v in run.data.params.items()])
    # return param_config
    # except Exception:
    try:
        config_path = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id,
            artifact_path=config_file,
            tracking_uri=tracking_uri,
        )
        config = config_from_path(config_path)

        if new_root is not None:
            resolve_paths(config, new_root)
        return config
    except Exception as e:
        raise Exception(f"Error loading config for run ID: {run.info.run_id} - {e}")


def get_latest(
    tracking_uri: str,
    *,
    experiment_name=None,
    run_name=None,
    run_id=None,
    filter_string=None,
    registry_uri=None,
    new_root=None,
):
    run: Run | str | None = None
    if run_name is not None:
        lookup = ("name", run_name)
        runs = list(
            search_runs(
                tracking_uri, registry_uri, max_results=1, experiment_name=experiment_name, filter_string=filter_string
            )
        )
        if len(runs) == 1:
            run = runs[0]
            print(f"Found run: {run.info.run_name} ({run.info.run_id})")

    elif run_id is not None:
        lookup = ("id", run_id)
        run = run_id
    else:
        raise Exception("One of run_name or run_id must be specified")

    if run is None:
        raise Exception(f"Unknown run {lookup[0]}: {lookup[1]}")

    config, model = load_ckpt_from_run(
        run,
        tracking_uri,
        new_root=new_root,
    )
    return config, model


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
