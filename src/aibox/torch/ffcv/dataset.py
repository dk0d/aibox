from torch.utils.data import Dataset
from pathlib import Path
from aibox.torch.ffcv.beton import create_beton_wrapper
from aibox.config import Config, init_from_cfg
from aibox.utils import as_path
from ffcv.loader import OrderOption
from ffcv.reader import Reader


class FFCVDataset(Dataset):
    def __init__(
        self,
        file_path: Path | str,
        pipeline_transforms: list[list],
        ordering: OrderOption = OrderOption.SEQUENTIAL,
        *,
        dataset_config: Config,
        fields: tuple | None = None,
        page_size: int | None = None,
        num_workers: int = -1,
        indices: list[int] | None = None,
        chunksize: int = 100,
        shuffle_indices: bool = False,
        verbose=False,
        **kwargs,
    ):
        """


        Args:
            file_path: path to the .beton file that needs to be loaded.
            pipeline_transforms: similar to FFCV Pipelines: https://docs.ffcv.io/making_dataloaders.html
                each item is a list of operations to perform on the specific object returned by the
                dataset. Note that order matters.
                If one item is None, will apply the default pipeline.
            ordering: order option for this pipeline, following FFCV specs on Dataset Ordering.
        """

        super().__init__()
        self.dataset = init_from_cfg(dataset_config)
        self.file_path = as_path(file_path)
        if not self.file_path.exists():
            create_beton_wrapper(
                self.dataset,
                str(self.file_path),
                fields=fields,
                page_size=page_size,
                num_workers=num_workers,
                indices=indices,
                chunksize=chunksize,
                shuffle_indices=shuffle_indices,
                verbose=verbose,
                **kwargs,
            )
        self.ordering = ordering

        self.pipeline = {}
        field_names = Reader(file_path).field_names

        if len(field_names) != len(pipeline_transforms):
            raise AttributeError(
                f"Passed pipeline_transforms object must include transforms for {len(field_names)} "
                f"items, {len(pipeline_transforms)} specified."
            )

        for name, transforms in zip(field_names, pipeline_transforms):
            if transforms is not None:
                self.pipeline[name] = transforms
