import multiprocessing as mp
from aibox.config import Config, init_from_cfg
from aibox.ffcv.dataset import FFCVDataset
from aibox.logger import get_logger


LOGGER = get_logger(__name__)

try:
    import lightning as L
    from ffcv.loader import Loader

    class FFCVDataModule(L.LightningDataModule):
        def __init__(
            self,
            batch_size: int,
            is_distributed: bool = False,
            num_workers: int | None = None,
            train_dataset: FFCVDataset | None = None,
            val_dataset: FFCVDataset | None = None,
            test_dataset: FFCVDataset | None = None,
            predict_dataset: FFCVDataset | None = None,
            os_cache: bool = True,
            seed: int | None = None,
            **kwargs,
        ) -> None:
            """
            Define PL DataModule (https://lightning.ai/docs/pytorch/stable/data/datamodule.html) object using
            FFCV Loader (https://docs.ffcv.io/making_dataloaders.html)

            Args:
                batch_size: batch_size for loader objects
                num_workers: num workers for loader objects
                is_distributed: pass true if using more than one gpu/node
                train_dataset: dataset for the training data, ignore if not loading train data
                val_dataset: dataset for the validation data, ignore if not loading validation data
                test_dataset: dataset for the test data, ignore if not loading test data
                predict_dataset: dataset for the predict data, ignore if not loading predict data
                os_cache: option for the ffcv loader, depending on your dataset.
                    Read official docs: https://docs.ffcv.io/parameter_tuning.html
                seed: fix data loading process to ensure reproducibility
                kwargs: pass any extra argument of the FFCV Loader object using the format "type_pname", where type is
                    one of {train, val, test, predict} and type is one of {indices, custom_fields, drop_last, batches_ahead,
                    recompile}. Check out https://docs.ffcv.io/making_dataloaders.html for more information about the parameters.
            """

            # initial condition must be satisfied
            if train_dataset is None and val_dataset is None and test_dataset is None and predict_dataset is None:
                raise AttributeError("At least one file between train, val, test and predict dataset must be specified")

            super().__init__()

            self.batch_size = batch_size
            self.num_workers = num_workers if num_workers is not None else mp.cpu_count()
            self.seed = seed
            self.os_cache = os_cache
            self.is_dist = is_distributed
            self.kwargs = kwargs

            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.test_dataset = test_dataset
            self.predict_dataset = predict_dataset

        def prepare_data(self) -> None:
            """
            This method is used to define the processes that are meant to be performed by only one GPU.
            It’s usually used to handle the task of downloading the data.
            """
            pass

        def setup(self, stage: str) -> None:
            pass

        def get_split_kwargs(self, split):
            keys = [
                ("indices", None),
                ("custom_fields", {}),
                ("drop_last", True),
                ("batches_ahead", 3),
                ("recompile", False),
            ]
            return {k: self.kwargs.get(f"{split}_{k}", default) for k, default in keys}

        def train_dataloader(self):
            if self.train_dataset is not None:
                return Loader(
                    self.train_dataset.file_path.as_posix(),
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    os_cache=self.os_cache,
                    order=self.train_dataset.ordering,  # type: ignore
                    pipelines=self.train_dataset.pipeline,
                    distributed=self.is_dist,
                    seed=self.seed,  # type: ignore
                    **self.get_split_kwargs("train"),
                )

        def val_dataloader(self):
            if self.val_dataset is not None:
                return Loader(
                    self.val_dataset.file_path.as_posix(),
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    os_cache=self.os_cache,
                    order=self.val_dataset.ordering,  # type: ignore
                    pipelines=self.val_dataset.pipeline,
                    distributed=self.is_dist,
                    seed=self.seed,  # type: ignore
                    **self.get_split_kwargs("val"),
                )

        def test_dataloader(self):
            if self.test_dataset is not None:
                return Loader(
                    self.test_dataset.file_path.as_posix(),
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    os_cache=self.os_cache,
                    order=self.test_dataset.ordering,  # type: ignore
                    pipelines=self.test_dataset.pipeline,
                    distributed=self.is_dist,
                    seed=self.seed,  # type: ignore
                    **self.get_split_kwargs("test"),
                )

        def predict_dataloader(self):
            if self.predict_dataset is not None:
                return Loader(
                    self.predict_dataset.file_path.as_posix(),
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    os_cache=self.os_cache,
                    order=self.predict_dataset.ordering,  # type: ignore
                    pipelines=self.predict_dataset.pipeline,
                    distributed=self.is_dist,
                    seed=self.seed,  # type: ignore
                    **self.get_split_kwargs("predict"),
                )

    class FFCVDataModuleFromConfig(FFCVDataModule):
        def __init__(
            self,
            *,
            train_dataset: Config | None = None,
            val_dataset: Config | None = None,
            test_dataset: Config | None = None,
            predict_dataset: Config | None = None,
            **kwargs,
        ):
            # Assumed to be FFCVDatasetFromConfigs
            if train_dataset is not None:
                kwargs.update(train_dataset=init_from_cfg(train_dataset))
            if val_dataset is not None:
                kwargs.update(val_dataset=init_from_cfg(val_dataset))
            if test_dataset is not None:
                kwargs.update(test_dataset=init_from_cfg(test_dataset))
            if predict_dataset is not None:
                kwargs.update(predict_dataset=init_from_cfg(predict_dataset))

            super().__init__(**kwargs)

except ImportError:
    import sys

    LOGGER.error("FFCV or FFCV Dependencies are not installed. Please see https://ffcv.io for more.")
    sys.exit(0)
