import argparse
from pathlib import Path

import aibox
import lightning as L
from lightning.fabric.utilities.exceptions import MisconfigurationException
import polars as pl
import pyarrow.parquet as pq
import torch
from aibox.config import Config, init_from_cfg
from aibox.logger import get_logger
from aibox.mlflow import MLFlowHelper
from aibox.progress import track
from aibox.torch.logging import CombinedLogger
from aibox.utils import as_path
from ffcv.loader import Loader
from torch.utils.data import DataLoader

LOGGER = get_logger(__name__)


class Evaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        loaders: list[tuple[Loader | DataLoader, dict]],
    ):
        self.model = model
        self.loaders = loaders

    def __len__(self):
        return sum([len(loader) for loader, _ in self.loaders])

    def evaluate(self, batch, meta, loader_idx, device=None) -> pl.DataFrame:
        raise NotImplementedError

    def __iter__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        with torch.no_grad():
            for i, (loader, meta) in enumerate(self.loaders):
                for batch in loader:
                    yield self.evaluate(batch, meta, i, device=device)


class ParquetWriter:
    """Save results to parquet file and log file with logger"""

    def __init__(
        self,
        evaluator: Evaluator,
        logger: CombinedLogger,
        results_dir: Path | None = None,
        cleanup_cache: bool = False,
        split="test",
    ):
        """Initialize the ParquetWriter


        Args:
            evaluator (Evaluator): Evaluator object. must be an iterable that returns a polars.DataFrame
                each iteration for each batch with the results of the evaluation.
            logger (CombinedLogger): The logger which will log the resulting parquet file to the mlflow run.
            results_dir (Path | None, optional): results directory where the parquet files will be written before being logged. Defaults to None.
            cleanup_cache (bool, optional): Delete the parquet file after logging. Defaults to False.
            split (str, optional): Split used in filename. Defaults to "test".
                Defaults to None.

        """
        self.evaluator = evaluator
        self.logger = logger
        self.split = split
        self.results_dir = results_dir if results_dir is not None else as_path("./results")

        filename = f"{self.logger.run_name}_{split}"
        self.parquet_path = self.results_dir / logger.run_name / f"{filename}.parquet"
        self.parquet_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure to not overwrite existing files of same name
        i = 0
        while self.parquet_path.exists():
            i += 1
            self.parquet_path = self.parquet_path.with_stem(f"{filename}_{i}")

        self.delete_after = cleanup_cache
        self.pq_writer = None

    def run(self):
        run_name = self.logger.run_name

        for df in track(self.evaluator, total=len(self.evaluator), description=f"{self.logger.run_name}"):
            df = df.with_columns(
                [
                    pl.lit(run_name).alias("run"),
                    pl.lit(self.logger.run_id).alias("run_id"),
                ],
            )
            df_arrow = df.to_arrow()
            if self.pq_writer is None:
                self.pq_writer = pq.ParquetWriter(self.parquet_path, df_arrow.schema)
            self.pq_writer.write_table(df_arrow)

        self.close_writer()

        self.logger.log_artifact(str(self.parquet_path))
        if self.delete_after:
            self.parquet_path.unlink()

    def close_writer(self):
        if self.pq_writer is not None:
            self.pq_writer.close()
            self.pq_writer = None  # remove this writer so a new one can be created if needed


def _eval_model(
    config: Config,
    model: torch.nn.Module | L.LightningModule,
    loaders: list[tuple[Loader | DataLoader, dict]],
    split,
    logger: CombinedLogger,
):
    evaluator = init_from_cfg(
        config.evaluator[split],
        model=model,
        loaders=loaders,
    )
    writer = init_from_cfg(
        config.evaluator.writer,
        evaluator=evaluator,
        logger=logger,
        split=split,
    )
    writer.run()


def evaluate_model(
    config: Config,
    model: torch.nn.Module | L.LightningModule,
    logger: CombinedLogger,
    dm: L.LightningDataModule,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    try:
        LOGGER.info("Running evaluation on Test Split")
        _eval_model(config, model, dm.test_dataloader(), "test", logger)
        LOGGER.info("TEST DONE")
    except MisconfigurationException:
        LOGGER.info("No Test Dataloaders Found. Skipping Evaluate Test Split")

    try:
        LOGGER.info("Running evaluation on Predict Split")
        _eval_model(config, model, dm.predict_dataloader(), "predict", logger)
        LOGGER.info("PREDICT DONE")
    except MisconfigurationException:
        LOGGER.info("No Predict Dataloaders Found. Skipping Evaluate Predict Split.")


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mlflow-tracking-uri",
        "-uri",
        default="./logs/mlruns",
    )
    parser.add_argument(
        "-rid",
        "--run-id",
        default=None,
    )
    parser.add_argument(
        "-rname",
        "--run-name",
        default=None,
    )

    # TODO: add ability to specify experiment name or id and test the latest run
    # parser.add_argument("-eid", "--experiment-id", default=None)
    # parser.add_argument("-ename", "--experiment-name", default=None)

    args = parser.parse_args(args)

    if args.run_id is None and args.run_name is None:
        LOGGER.info('One of "run-id" or "run-name" must be given')
        exit(0)

    # if args.experiment_name is not None:
    #     experiment = client.get_experiment_by_name(args.experiment_name)
    # else:
    #     experiment = client.get_experiment(args.experiment_id)

    if ":" not in args.mlflow_tracking_uri:  # FIXME: this is a hack check
        uri = f"file://{Path(args.mlflow_tracking_uri).expanduser().resolve()}"
    else:
        uri = args.mlflow_tracking_uri

    helper = MLFlowHelper(tracking_uri=uri)

    if args.run_name is not None:
        lookup = ("name", args.run_name)
        try:
            run = helper.search_runs(args.run_name)[0]
            LOGGER.info(f"Found run: {run.info.run_name} ({run.info.run_id})")
        except IndexError:
            run = None
    else:
        lookup = ("id", args.run_id)
        run = helper.get_run(args.run_id)

    if run is None:
        LOGGER.error(f"Unknown run {lookup[0]}: {lookup[1]}")
        exit(1)

    config = helper.load_config_from_run(run)
    if config is None:
        LOGGER.error("No config found in run")
        exit(1)
    # LOGGER.info(config)

    # load the model
    _, model = helper.load_ckpt_from_run(run)
    if model is None:
        exit(1)

    LOGGER.info(f"Loaded Checkpoint for Model: {model.__class__.__name__}")

    # init the data module
    dm = init_from_cfg(config.data)

    LOGGER.info(f"Loaded DataModule: {dm.__class__.__name__}")
    if isinstance(dm, aibox.torch.lightning.DataModuleFromConfig):
        for split, conf in dm.dataset_configs.items():
            LOGGER.info(f"{split}: {conf['__classpath__']}")

    # Get logger pointing to the run
    logger = CombinedLogger(
        tracking_uri=helper.tracking_uri,
        run_name=run.info.run_name,
        run_id=run.info.run_id,
    )

    # Run evaluate
    evaluate_model(config, model, logger, dm)
