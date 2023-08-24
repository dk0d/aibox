# %%
import os

import lightning as L
import torch
from lightning import seed_everything
from lightning.pytorch.loggers import Logger
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning.pytorch.callbacks import RichProgressBar

from aibox.cli import AIBoxCLI, OmegaConf
from aibox.config import init_from_cfg
from aibox.logger import get_logger

EVALUATE_OUTPUT = list[dict[str, float]]  # 1 dict per DataLoader


LOGGER = get_logger(__name__)


def init_callbacks(config) -> list[L.Callback]:
    """
    Helper function to initialize callbacks from config.

    Args:
        config: The config object.

    Returns:
        A list of callbacks.
    """
    if "callbacks" in config.trainer:
        callbacks_cfg = config.trainer.callbacks
    else:
        callbacks_cfg = OmegaConf.create()
    callbacks = [cfg for _, cfg in callbacks_cfg.items() if "__disable__" not in cfg or not cfg.__disable__]
    [cfg.pop("__disable__", None) for cfg in callbacks]
    callbacks = [init_from_cfg(cfg) for cfg in callbacks]
    callbacks.append(RichProgressBar())
    return callbacks


def init_logger(config, log_hyperparams=True) -> Logger | None:
    """
    Helper function to initialize logger from config.

    Args:
        config: The config object.

    Returns:
        A logger.
    """
    if "logging" in config:
        try:
            logger = init_from_cfg(config.logging)
            if "tags" in config and hasattr(logger, "log_tags"):
                logger.log_tags(config.tags)
                del config.tags
            if log_hyperparams:
                logger.log_hyperparams(config)

            if hasattr(logger, "experiment_name"):
                LOGGER.info(f"Experiment Name: {logger.experiment_name}")
            if hasattr(logger, "run_id"):
                LOGGER.info(f"Run ID: {logger.run_id}")
            return logger
        except Exception as e:  # TODO: Better handling of when logger init fails
            LOGGER.error(f"Failed to initialize logger: {config.logging}")
            exit(1)
    return None


def handle_slurm_param(config):
    """
    Helper function to handle slurm parameters.

    Note:
        This function is not used in the current version of the code.

    Args:
        config: The config object.

    Returns:
        A dictionary of slurm parameters.
    """
    slurm = "SLURM_JOB_ID" in os.environ.keys()
    if slurm:
        slurmMeta = {
            k: os.environ.get(k, None)
            for k in [
                "SLURM_JOB_ID",
                "SLURM_JOB_NODELIST",
                "SLURM_JOB_NUM_NODES",
                "SLURM_NTASKS",
                "SLURM_TASKS_PER_NODE",
                "SLURM_MEM_PER_NODE",
                "SLURM_MEM_PER_CPU",
                "SLURM_NODEID",
                "SLURM_PROCID",
                "SLURM_LOCALID",
                "SLURM_TASK_PID",
                "SLURM_CPUS_ON_NODE",
            ]
        }

        try:
            config.data.num_workers = int(slurmMeta["SLURM_TASKS_PER_NODE"])
        except Exception as e:
            LOGGER.error("Failed to set num_workers from SLURM_TASKS_PER_NODE", exc_info=True)
        # torch.set_float32_matmul_precision("medium")
    else:
        slurmMeta = {}

    return slurmMeta


def init_trainer(config):
    """
    Helper function to initialize trainer from config.

    Args:
        config: The config object.

    Returns:
        A trainer.
    """
    trainerParams = dict(**config.trainer)

    logger = init_logger(config)

    if logger is not None:
        LOGGER.info(f"Logging to: {logger.log_dir}")
        trainerParams.update(logger=logger)

    callbacks = init_callbacks(config)

    try:
        if "tuner" in config and config.tuner.mode == "ray":
            from ray.tune.integration.lightning import TuneReportCallback

            tune_callback = TuneReportCallback(
                metrics={"loss": "val/loss", "acc": "val/acc"},
                on="validation_end",
            )
            callbacks.append(tune_callback)

            trainerParams.update(enable_progress_bar=False)
    except Exception as e:
        LOGGER.error("error in init_trainer", exc_info=True)
        pass

    strategy = "auto"

    # Set accelerator
    if torch.has_mps:
        accelerator = "mps"
    else:
        accelerator = "gpu"

        if "strategy" in config.trainer:
            strategy = init_from_cfg(config.trainer.strategy)
        else:
            if torch.has_cuda and torch.cuda.device_count() > 1:
                strategy = DDPStrategy(find_unused_parameters=False)

    if "profiler" in config.trainer:
        profiler = init_from_cfg(config.trainer.profiler)
    else:
        profiler = None

    trainerParams.update(
        strategy=strategy,
        accelerator=accelerator,
        devices="auto",
        num_sanity_val_steps=0,
        fast_dev_run=config.debug,
        callbacks=callbacks,
        precision=32,
        profiler=profiler,
    )

    if not torch.has_cuda and not torch.has_mps:
        trainerParams.pop("accelerator")
        trainerParams.pop("devices")
        trainerParams.pop("strategy")

    trainer = L.Trainer(**trainerParams)
    return trainer


def init_model(config) -> L.LightningModule:
    """
    Helper function to initialize model from config.

    Args:
        config: The config object.

    Returns:
        A model.
    """
    model = init_from_cfg(config.model)

    try:
        # TODO: Add support for logging model graph in combined logger
        if config.logging.args.log_graph:
            if not hasattr(model, "example_input_array"):
                LOGGER.info("[yellow bold] WARNING [/yellow bold]: Model does not `example_input_array`")
                LOGGER.info("[yellow bold italic]Model must have `example_input_array` attribute when log_graph=True")
                LOGGER.info("[blue] Setting log_graph=False")
                config.logging.log_graph = False
    except Exception:
        pass
    return model


def train(config) -> tuple[L.LightningModule, L.LightningDataModule, L.Trainer]:
    """
    Run training from config.

    Args:
        config: The config object.

    Returns:
        A tuple of model, datamodule, and trainer.
    """

    trainer = init_trainer(config)

    # Seed
    # Make sure to do this after logging is initialized
    # (Otherwise MLFlow will generate the same run name for different runs)
    # and before initializing model
    if "seed" in config:
        seed_everything(config.seed, True)

    model = init_model(config)
    LOGGER.info(f"MODEL INITIALIZED: {model.__class__.__name__}")

    dm = init_from_cfg(config.data)
    LOGGER.info(f"DATAMODULE INITIALIZED: {dm.__class__.__name__}")

    LOGGER.info("TRAINING START")
    trainer.fit(model=model, datamodule=dm)
    LOGGER.info("TRAINING DONE")

    return model, dm, trainer


def train_and_test(config) -> EVALUATE_OUTPUT | None:
    """
    Run training and testing from config.

    Args:
        config: The config object.

    Returns:
        A tuple of model, datamodule, and trainer.
    """
    model, dm, trainer = train(config)

    if not is_overridden("test_step", model):
        LOGGER.info("No testing step found on model. Skipping testing")
        return None
    try:
        LOGGER.info("TESTING START")
        testing_results = trainer.test(model=model, datamodule=dm)
        LOGGER.info("TESTING DONE")
        return testing_results
    except Exception:
        LOGGER.exception("error during test")
        pass

    return None


def main(args=None):
    """
    Main function to run training and testing from config.

    Args:
        args: The arguments to be parsed.

    Returns:
        Results of testings
    """
    cli = build_cli_parser()

    config = cli.parse_args(args=args)

    if config.name is None:
        LOGGER.error("Experiment name must be specified")
        exit(1)

    handle_slurm_param(config)
    results = train_and_test(config)

    return results


def build_cli_parser(args=None):
    cli = AIBoxCLI()
    # First key takes priority
    # cli.add_linked_properties("data.args.img_size", "model.args.img_size", default=64)
    # cli.add_linked_properties("data.args.n_channels", "model.args.in_channels", default=3)
    return cli


if __name__ == "__main__":
    main()
