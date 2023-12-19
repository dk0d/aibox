from pathlib import Path

from ffcv.loader import OrderOption
from ffcv.reader import Reader
from torch.utils.data import Dataset

from aibox.config import Config, init_from_cfg
from aibox.ffcv.beton import create_beton_wrapper
from aibox.utils import as_path


class FFCVDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        file_path: Path | str,
        pipeline_transforms: list[list],
        ordering: OrderOption = OrderOption.SEQUENTIAL,
        *,
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

        Wraps a torch dataset and creates a beton file for it if it does not exist.


        Args:
            file_path: path to the .beton file that needs to be loaded.
            pipeline_transforms: similar to FFCV Pipelines: https://docs.ffcv.io/making_dataloaders.html
                each item is a list of operations to perform on the specific object returned by the
                dataset. Note that order matters.
                If one item is None, will apply the default pipeline.
            ordering: order option for this pipeline, following FFCV specs on Dataset Ordering.
            fields: use this if you want to use Fields different from default.
                The length and order must respect the "get_item" return of the torch Dataset.
                If you want to overwrite only some fields, pass None to the remaining positions.
            page_size: page size internally used.
                (optional argument of DatasetWriter object)
            num_workers: Number of processes to use.
                (optional argument of DatasetWriter object)
            indices: Use a subset of the dataset specified by indices.
                (optional argument of from_indexed_dataset method)
            chunksize: Size of chunks processed by each worker during conversion.
                (optional argument of from_indexed_dataset method)
            shuffle_indices: Shuffle order of the dataset.
                (optional argument of from_indexed_dataset method)
        """

        super().__init__()
        self.dataset = dataset
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


class FFCVDatasetFromConfig(FFCVDataset):
    def __init__(self, dataset_config: Config, **kwargs):
        super().__init__(dataset=init_from_cfg(dataset_config), **kwargs)
