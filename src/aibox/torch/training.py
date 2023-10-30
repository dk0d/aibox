# %%
import os
from typing import Any

import lightning as L
import torch
from lightning import seed_everything
from lightning.pytorch.callbacks import Callback, RichProgressBar
from lightning.pytorch.loggers import Logger
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.utilities.model_helpers import is_overridden

from aibox.cli import AIBoxCLI, OmegaConf
from aibox.config import config_update, init_from_cfg
from aibox.logger import get_logger

try:
    from lightning.pytorch.plugins.environments import LightningEnvironment  # type: ignore
    from ray import train as ray_train
    from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
    from ray.train.lightning import RayLightningEnvironment, RayDDPStrategy
    from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
    from ray.train.lightning._lightning_utils import get_worker_root_device

    class RayDDPStrategyWrapper(DDPStrategy, RayDDPStrategy):
        """Subclass of DDPStrategy to ensure compatibility with Ray orchestration.

        For a full list of initialization arguments, please refer to:
        https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.strategies.DDPStrategy.html
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            record_extra_usage_tag(TagKey.TRAIN_LIGHTNING_RAYDDPSTRATEGY, "1")  # type: ignore

        @property
        def root_device(self) -> torch.device:
            return get_worker_root_device()

        @property
        def distributed_sampler_kwargs(self) -> dict[str, Any]:
            return dict(
                num_replicas=self.world_size,
                rank=self.global_rank,
            )

    class RayLightningEnvironmentWrapper(LightningEnvironment, RayLightningEnvironment):
        """Setup Lightning DDP training environment for Ray cluster."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            record_extra_usage_tag(TagKey.TRAIN_LIGHTNING_RAYLIGHTNINGENVIRONMENT, "1")  # type: ignore

        def world_size(self) -> int:
            return ray_train.get_context().get_world_size() or 1

        def global_rank(self) -> int:
            return ray_train.get_context().get_world_rank() or 0

        def local_rank(self) -> int:
            return ray_train.get_context().get_local_rank() or 0

        def node_rank(self) -> int:
            return ray_train.get_context().get_node_rank() or 0

        def set_world_size(self, size: int) -> None:
            # Disable it since `world_size()` directly returns data from AIR session.
            pass

        def set_global_rank(self, rank: int) -> None:
            # Disable it since `global_rank()` directly returns data from AIR session.
            pass

        def teardown(self):
            pass

    class RayTuneReportCheckpointCallback(Callback, TuneReportCheckpointCallback):
        pass

    class RayTrainReportCallback(Callback):
        """A simple callback that reports checkpoints to Ray on train epoch end."""

        def __init__(self) -> None:
            super().__init__()

            self.trial_name = ray_train.get_context().get_trial_name()
            self.local_rank = ray_train.get_context().get_local_rank()
            # self.tmpdir_prefix = os.path.join(tempfile.gettempdir(), self.trial_name)
            # if os.path.isdir(self.tmpdir_prefix) and self.local_rank == 0:
            #     shutil.rmtree(self.tmpdir_prefix)

            record_extra_usage_tag(TagKey.TRAIN_LIGHTNING_RAYTRAINREPORTCALLBACK, "1")  # type: ignore

        def on_validation_epoch_end(self, trainer, pl_module) -> None:
            # # Creates a checkpoint dir with fixed name
            # tmpdir = os.path.join(self.tmpdir_prefix, str(trainer.current_epoch))
            # os.makedirs(tmpdir, exist_ok=True)

            # Fetch metrics
            metrics = trainer.callback_metrics
            metrics = {k: v.item() for k, v in metrics.items()}

            # (Optional) Add customized metrics
            metrics["epoch"] = trainer.current_epoch
            metrics["step"] = trainer.global_step

            # Save checkpoint to local
            # ckpt_path = os.path.join(tmpdir, "checkpoint.ckpt")
            # trainer.save_checkpoint(ckpt_path, weights_only=False)

            # Report to train session
            # checkpoint = ray_train.Checkpoint.from_directory(tmpdir)
            # ray_train.report(metrics=metrics, checkpoint=checkpoint)
            ray_train.report(metrics=metrics)

            # if self.local_rank == 0:
            #     shutil.rmtree(tmpdir)

except ImportError:
    pass

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
            if hasattr(logger, "run_name"):
                LOGGER.info(f"Run Name: {logger.run_name}")
            return logger
        except Exception as e:  # TODO: Better handling of when logger init fails
            LOGGER.error(f"Failed to initialize logger: {config.logging}")
            LOGGER.error(e, exc_info=True)
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
                "SLURM_TASKS_PER_CPU",
                "SLURM_MEM_PER_NODE",
                "SLURM_MEM_PER_CPU",
                "SLURM_NODEID",
                "SLURM_PROCID",
                "SLURM_LOCALID",
                "SLURM_TASK_PID",
                "SLURM_CPUS_PER_TASK",
            ]
        }

        try:
            for k in ["SLURM_TASKS_PER_NODE", "SLURM_CPUS_PER_TASK"]:
                n = slurmMeta[k]
                if n is not None:
                    config_update(config, "data.num_workers", int(n))
                    break
        except Exception as e:
            LOGGER.error("Failed to set num_workers from SLURM_TASKS_PER_NODE", exc_info=True)
            LOGGER.exception(e)
        # torch.set_float32_matmul_precision("medium")
    else:
        slurmMeta = {}

    return slurmMeta


def init_trainer(config, **kwargs):
    """
    Helper function to initialize trainer from config.

    Args:
        config: The config object.

    Returns:
        A trainer.
    """

    handle_slurm_param(config)

    trainerParams = dict(**config.trainer)

    if "fast_dev_run" not in trainerParams:
        trainerParams.update(fast_dev_run=config.debug)

    if kwargs.get("should_init_logger", True):
        logger = init_logger(config)
        if logger is not None:
            LOGGER.info(f"Logging to: {logger.log_dir}")
            trainerParams.update(logger=logger)

    callbacks = init_callbacks(config)

    # Remove progress bar if disabled -- raises error otherwise
    if "enable_progress_bar" in trainerParams and not trainerParams["enable_progress_bar"]:
        callbacks = [cb for cb in callbacks if not isinstance(cb, RichProgressBar)]

    LOGGER.info(f"Callbacks: {[c.__class__.__name__.split('.')[-1] for c in callbacks]}")

    strategy = "auto"

    # Set accelerator
    if torch.has_mps:
        accelerator = "mps"
    else:
        accelerator = "gpu"

        if "strategy" in config.trainer:
            strategy = init_from_cfg(config.trainer.strategy)
        else:
            if "tuner" in config:
                accelerator = "auto"
                strategy = RayDDPStrategyWrapper(find_unused_parameters=False)
                trainerParams.update(plugins=[RayLightningEnvironmentWrapper()])
            elif torch.has_cuda and torch.cuda.device_count() > 1:
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
        callbacks=callbacks,
        precision=32,
        profiler=profiler,
    )

    if not torch.has_cuda and not torch.has_mps:
        trainerParams.pop("accelerator")
        trainerParams.pop("devices")
        trainerParams.pop("strategy")

    trainer = L.Trainer(**trainerParams)
    if "tuner" in config:
        from ray.train.lightning import prepare_trainer

        trainer = prepare_trainer(trainer)  # type: ignore
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


def train(config, **kwargs) -> tuple[L.LightningModule, L.LightningDataModule, L.Trainer]:
    """
    Run training from config.

    Args:
        config: The config object.

    Returns:
        A tuple of model, datamodule, and trainer.
    """

    trainer = init_trainer(config, **kwargs)

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


def train_and_test(config, **kwargs) -> EVALUATE_OUTPUT | None:
    """
    Run training and testing from config.

    Args:
        config: The config object.

    Returns:
        A tuple of model, datamodule, and trainer.
    """

    model, dm, trainer = train(config, **kwargs)

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
