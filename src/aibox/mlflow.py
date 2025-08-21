from dataclasses import dataclass
from pathlib import Path
import functools

import lightning as L
from mlflow import artifacts
import yaml

import mlflow
from aibox.config import (
    Config,
    class_from_string,
    config_from_path,
    derive_args,
    derive_classpath,
    resolve_paths,
)
from aibox.logger import get_logger
from aibox.torch.utils import get_device
from aibox.utils import as_path
from mlflow.entities import Run, experiment
from mlflow.store.entities.paged_list import PagedList

LOGGER = get_logger(__name__)


class MLFlowHelper:
    def __init__(
        self,
        tracking_uri,
        registry_uri=None,
    ):
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri
        self.client = mlflow.MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)

    def get_run(self, run_id):
        """Get a run by its run_id"""
        return self.client.get_run(run_id=run_id)

    def experiments(self):
        """Get a list of all experiments in the MLFlow tracking server"""
        return [e for e in self.client.search_experiments() if e.name != "Default"]

    def experiment_by_name(self, name):
        """Get an experiment by its name"""
        return self.client.get_experiment_by_name(name)

    def experiment_by_id(self, id):
        """Get an experiment by its id"""
        return self.client.get_experiment(id)

    def get_runs(
        self,
        run_name=None,
        experiment_ids: list[str] | None = None,
        max_results=1000,
        filter_string: str | None = None,
        order_by: list[str] | None = None,
        page_token: str | None = None,  # get from paged list
    ) -> PagedList[Run]:
        """Get runs in MLFlow"""
        runs = self.search_runs(
            run_name=run_name,
            experiment_ids=experiment_ids,
            max_results=max_results,
            filter_string=filter_string,
            order_by=order_by,
            page_token=page_token,
        )
        return runs

    def get_latest(
        self,
        experiment_name: str | None = None,
        run_name: str | None = None,
        run_id: str | None = None,
        filter_string: str | None = None,
        init_model: bool = True,
        new_root: str | Path | None = None,
        return_run: bool = False,
    ) -> tuple[Config, L.LightningModule] | tuple[Config, L.LightningModule, Run]:
        """
        Get the latest run from MLFlow and loads the associated config and model given
        the experiment name, run_name, or run_id, or some combination thereof.

        An MLFlow `filter_string` can be used to filter runs.
        See []() for more

        Args:
            experiment_name (str, optional): [description]. Defaults to None.
            run_name (str, optional): [description]. Defaults to None.
            run_id (str, optional): [description]. Defaults to None.
            filter_string (str, optional): [description]. Defaults to None.
            registry_uri (str, optional): [description]. Defaults to None.
            init_model (bool): Whether to initialize the model or just return the ckpt file. Defaults to True.
            new_root (str | Path, optional): [description]. Defaults to None.
            return_run (bool): whether to return the run object. Defaults to False.

        Returns:
            tuple[Config, L.LightningModule]: configuration and model used in run
        """
        run: Run | str | None = None
        if run_name is not None:
            lookup = ("name", run_name)
            runs = list(
                self.search_runs(
                    max_results=1,
                    run_name=run_name,
                    experiment_name=experiment_name,
                    filter_string=filter_string,
                )
            )
            if len(runs) == 1:
                run = runs[0]
                LOGGER.info(f"Found run: {run.info.run_name} ({run.info.run_id})")
            else:
                LOGGER.warning(f"Found more than 1 run with {lookup[0]}: {lookup[1]}")

        elif run_id is not None:
            lookup = ("id", run_id)
            run = run_id
        else:
            msg = "One of run_name or run_id must be specified"
            LOGGER.exception(msg)
            raise Exception(msg)

        if run is None:
            raise Exception(f"Unknown run {lookup[0]}: {lookup[1]}")

        config, model = self.load_ckpt_from_run(
            run,
            init_model=init_model,
            new_root=new_root,
        )
        if return_run:
            if isinstance(run, str):
                run = mlflow.get_run(run)
            return config, model, run
        return config, model

    def load_ckpt_from_run(
        self,
        run: Run | str,
        alias="best",
        new_root=None,
        init_model=True,
        verbose=False,
        path_keys: list[str] | None = None,
        **model_kwargs,
    ) -> tuple[Config, L.LightningModule]:
        """
        Loads the model from the run_id or run object

        Args:
            run (Run | str): run object or `run_id`
            alias (str, optional): [description]. Defaults to "best".
            new_root ([type], optional): [description]. Defaults to None.
            init_model (bool, optional): whether or not to call `load_from_checkpoint()`. If False,
                will just load the raw ckpt file with `torch.load()`
                Defaults to True.
            path_keys (list[str], optional)
            **model_kwargs: [description]. extra params passed to model initializer. Defaults to {}.

        Returns:
            tuple[Config, L.LightningModule]: the configuration and the model used in run
        """
        if isinstance(run, str):  # is run_id
            run = self.client.get_run(run_id=run)

        model_dir = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id,
            artifact_path="model",
            tracking_uri=self.tracking_uri,
        )
        config = self.load_config_from_run(
            run,
            new_root=new_root,
            path_keys=path_keys,
            verbose=verbose,
        )
        checkpoints = get_mlflow_checkpoints(Path(model_dir))
        best = [c.path for c in checkpoints if alias in c.aliases][-1]
        # try:
        if init_model:
            if not hasattr(config, "model"):
                raise Exception("Config has no model key")
            model_cfg = config.model
            Model: L.LightningModule = class_from_string(derive_classpath(model_cfg))
            device = get_device()
            model_kwargs.update(map_location=device)
            model = Model.load_from_checkpoint(best, **derive_args(model_cfg, **model_kwargs))
        else:
            import torch

            model = torch.load(best)
        # except Exception as e:
        # raise Exception(f"Unable to load model for run: {run.info.run_id}\n{e}")
        return config, model

    def load_config_from_run(
        self,
        run: Run,
        config_file="config.yml",
        new_root=None,
        path_keys: list[str] | None = None,
        verbose=False,
    ):
        """
        Load the configuration from the run_id or run object

        Args:

            run (Run | str): run object or `run_id`
            config_file (str, optional): [description]. Defaults to "config.yml".
            new_root ([type], optional): If set, renames paths found in config files and artifact paths.
            Defaults to None.
            path_keys (list[str], optional)
            verbose (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        # try:
        # param_config = config_from_dotlist([f"{k}={v}" for k, v in run.data.params.items()])
        # return param_config
        # except Exception:
        try:
            config_path = mlflow.artifacts.download_artifacts(
                run_id=run.info.run_id,
                artifact_path=config_file,
                tracking_uri=self.tracking_uri,
            )
            config = config_from_path(config_path)

            if new_root is not None:
                resolve_paths(
                    config,
                    new_root,
                    path_keys=path_keys,
                    verbose=verbose,
                )
            return config
        except Exception as e:
            raise Exception(f"Error loading config for run ID: {run.info.run_id} - {e}")

    def search_experiments(
        self,
        max_results: int | None = 1000,
        filter_string: str | None = None,
        order_by: list[str] | None = None,
        page_token=None,
    ):
        return self.client.search_experiments(
            filter_string=filter_string,
            max_results=max_results,
            order_by=order_by,
            page_token=page_token,
        )

    def search_runs(
        self,
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
            run_name ([type], optional): [description]. Defaults to None.
            experiment_ids (list[str], optional): [description]. Defaults to None.
            max_results (int, optional): [description]. Defaults to 1.
            filter_string (str, optional): [description]. Defaults to None.
            order_by (list[str], optional): [description]. Defaults to None.
            page_token (str, optional): get from PagedList. Defaults to None.
        """

        if experiment_ids is None:
            exp_filter_string = None if experiment_name is None else f"attribute.name LIKE '%{experiment_name}%'"
            experiment_ids = [e.experiment_id for e in self.client.search_experiments(filter_string=exp_filter_string)]

        filters = [(f"attributes.run_name LIKE '%{run_name}%'" if run_name is not None else None)]

        if filter_string is not None:
            filters.append(filter_string)

        filters = [f for f in filters if f is not None]

        filter_string = " AND ".join(f for f in filters if f is not None) if len(filters) > 0 else ""

        order_by = order_by if order_by is not None else ["metrics.`test/loss` DESC"]

        runs = self.client.search_runs(
            experiment_ids,
            filter_string=filter_string,
            max_results=max_results,
            order_by=["metrics.`test/loss` DESC"],
            page_token=page_token,
        )

        return runs

    def log_artifact(self, run_id, local_path, artifact_path=None):
        self.client.log_artifact(run_id, local_path, artifact_path=artifact_path)

    def log_artifacts(self, run_id, local_dir, artifact_dir=None):
        self.client.log_artifacts(run_id, local_dir, artifact_path=artifact_dir)

    def get_artifact_paths(self, run: Run | str):
        run_id: str = run.info.run_id if isinstance(run, Run) else run
        return self.client.list_artifacts(run_id)

    def get_artifact(self, run: Run | str, path: str, dst_path: str | None = None) -> Path:
        run_id: str = run.info.run_id if isinstance(run, Run) else run
        return as_path(self.client.download_artifacts(run_id, path, dst_path))


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
    with open(metadata_file) as f:
        y = yaml.safe_load(f)
        curr = y[artifact_path_key]
        new = f"file://{pwd}"
        if curr == new:
            LOGGER.info(f"Skipping: {metadata_file}")
            return
        y[artifact_path_key] = new

    with open(metadata_file, "w") as f:
        # LOGGER.info(yaml.dump(y, default_flow_style=False, sort_keys=False))
        yaml.dump(y, f, default_flow_style=False, sort_keys=False)
        LOGGER.info(f"UPDATED {curr} -> {new}")


def fix_mlflow_artifact_paths(mlflow_root: Path):
    """
    Fixes the artifact paths in the metadata files for MLFlow runs and experiments.
    Rewrites artifact paths to be relative to the mlflow_root directory.
    Useful when moving MLFlow runs and experiments to a new location.

    """
    for experiment_folder in mlflow_root.expanduser().resolve().iterdir():
        metadata_file = experiment_folder / "meta.yaml"

        # Fix experiment metadata
        if metadata_file.exists():
            rewrite_artifact_path(metadata_file, experiment_folder, artifact_path_key="artifact_location")
        for run_folder in experiment_folder.iterdir():
            metadata_file = run_folder / "meta.yaml"
            # LOGGER.info(f"RUN FOLDER: {run_folder}")

            # Fix run metadata
            if metadata_file.exists():
                rewrite_artifact_path(
                    metadata_file,
                    run_folder / "artifacts",
                    artifact_path_key="artifact_uri",
                )


def fix_mlflow_artifact_uri_sqlite(mlflow_root: Path, db_path: str | Path, root_name="mlruns"):
    from sqlalchemy import create_engine, text

    db_path = as_path(db_path)

    if not db_path.exists():
        raise FileNotFoundError("SQLite DB not found")

    LOGGER.info(f"Found DB {db_path.name}")

    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as connection:
        LOGGER.info(f"Connected to {db_path.name}")
        experiments = connection.execute(text("select experiment_id, name, artifact_location from experiments;"))
        if experiments.rowcount >= 0:
            LOGGER.info(f"Found {experiments.rowcount} experiments")
        for exp in experiments:
            data = exp._asdict()
            uri = list(as_path(data["artifact_location"]).parts)

            try:
                idx = uri.index(root_name)
            except ValueError:
                LOGGER.error(f"Unable to update experiment {data['name']} - {root_name} not found in artifact_location")
                continue

            new = as_path(mlflow_root).joinpath(*uri[idx + 1 :])
            res = connection.execute(
                text(
                    "update experiments set artifact_location = :artifact_location where experiment_id = :experiment_id;"
                ).bindparams(experiment_id=f"{data['experiment_id']}", artifact_location=f"{new}"),
            )
            if res.rowcount > 0:
                LOGGER.info(f"Updated experiment {data['name']} ({data['experiment_id']}) to {new}")
            else:
                LOGGER.error(f"Unable to update experiment {data['name']}")

        runs = connection.execute(text("select run_uuid, name, artifact_uri from runs;"))

        if runs.rowcount >= 0:
            LOGGER.info(f"Found {runs.rowcount} runs")

        for exp in runs:
            data = exp._asdict()
            uri = list(as_path(data["artifact_uri"]).parts)
            try:
                idx = uri.index(root_name)
            except ValueError:
                LOGGER.error(f"Unable to update run {data['name']} - {root_name} not found in artifact_location")
                continue
            new = as_path(mlflow_root).joinpath(*uri[idx + 1 :])
            res = connection.execute(
                text("update runs set artifact_uri = :artifact_uri where run_uuid = :run_uuid;").bindparams(
                    run_uuid=f"{data['run_uuid']}", artifact_uri=f"{new}"
                ),
            )
            if res.rowcount > 0:
                LOGGER.info(f"Updated run {data['name']} ({data['run_uuid']}) to {new}")
            else:
                LOGGER.error(f"Unable to update run {data['name']}")

        connection.commit()

    LOGGER.info("Done")


def get_mlflow_run_metas(mlruns_root: Path):
    """Collects all of the metadata files for MLFlow runs in the mlruns directory."""
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
