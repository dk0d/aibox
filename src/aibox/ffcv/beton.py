from aibox.logger import get_logger

LOGGER = get_logger(__name__)

try:
    from pathlib import Path

    from ffcv import DatasetWriter
    from torch.utils.data import Dataset

    from aibox.ffcv.utils import field_to_str, obj_to_field
    from aibox.utils import as_path

    def create_beton_wrapper(
        torch_dataset: Dataset,
        output_path: str | Path,
        fields: tuple | None = None,
        page_size: int | None = None,
        num_workers: int = -1,
        indices: list[int] | None = None,
        chunksize: int = 100,
        shuffle_indices: bool = False,
        verbose=False,
    ) -> None:
        """
        Simple utility function that allows the creation of .beton files from Torch Datasets.
        References: https://docs.ffcv.io/writing_datasets.html

        The dataset can have any number/type of parameters in the __init__ method.

        Constraints on the __get_item__ method:
            According to the official ffcv docs, the dataset must return a tuple object of any length.

            The type of the elements inside the tuple is automatically mapped to the ffcv.fields admitted types:
            https://docs.ffcv.io/api/fields.html

        Default Fields (depending on objects obtained from Dataset.__get_item__):

        PIL Images - RGBImageField(writemode='jpg')

        Integers - IntField()

        Floats - FloatField()

        Numpy arrays - NDArrayField(obj.dtype, obj.shape)

        Dict - JSONField()

        Torch tensor - TorchTensorField(obj.dtype, obj.shape)

        1D uint8 numpy array - BytesField()

        Args:
            torch_dataset: Pytorch Dataset object (https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).
            output_path: desired path for .beton output file, "/" separated. E.g. "./my_dataset.beton"
            fields: use this if you want to use Fields different from default.
                The length and order must respect the "get_item" return of the torch Dataset.
                If you want to overwrite only some fields, pass None to the remaining positions.
            page_size: page size internally used. (optional argument of DatasetWriter object)
            num_workers: Number of processes to use. (optional argument of DatasetWriter object)
            indices: Use a subset of the dataset specified by indices. (optional argument of from_indexed_dataset method)
            chunksize: Size of chunks processed by each worker during conversion.
                (optional argument of from_indexed_dataset method)
            shuffle_indices: Shuffle order of the dataset. (optional argument of from_indexed_dataset method)
        """

        # get default page size (4 * MIN_PAGE_SIZE): --> https://github.com/libffcv/ffcv/blob/main/ffcv/writer.py
        page_size = 4 * (2**21) if page_size is None else page_size

        # 1. format output path
        # assert len(output_path) > 0, "param: output_path cannot be an empty string"

        output_path = as_path(output_path)

        if output_path.suffix == ".beton":
            output_path = output_path.with_suffix(".beton")

        # find dir
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            LOGGER.info(f"Creating output folder: {output_path.parent}")

        # 2. check that dataset __get_item__ returns a tuple and get fields.
        tuple_obj = torch_dataset[0]

        if not isinstance(tuple_obj, tuple):
            raise AttributeError(
                "According to the official ffcv docs, the dataset must return a tuple object. "
                "See for example: https://docs.ffcv.io/writing_datasets.html"
            )

        if fields is None:
            fields = tuple([None for _ in range(len(tuple_obj))])

        if not len(fields) == len(tuple_obj):
            raise AttributeError(
                "Passed a wrong number of 'fields' objects.\n"
                f"The __get_item__ method of the specified dataset returns {len(tuple_obj)} elements, but"
                f" {len(fields)} objects were passed."
            )

        final_fields = []
        for obj, f in zip(tuple_obj, fields):
            final_fields.append(obj_to_field(obj) if f is None else f)

        # 2. create dict of fields
        final_mapping = {}
        for i, f in enumerate(final_fields):
            final_mapping[f"{field_to_str(f)}_{i}"] = f

        # official guidelines: https://docs.ffcv.io/writing_datasets.html
        if verbose:
            LOGGER.info(f"creating ffcv dataset into file: {output_path}")
            LOGGER.info(f"number of items: {len(torch_dataset)}")  # type: ignore
            LOGGER.info(f"ffcv fields of items: {final_fields}")

        writer = DatasetWriter(
            output_path.as_posix(),
            final_mapping,
            page_size=page_size,
            num_workers=num_workers,
        )
        writer.from_indexed_dataset(
            torch_dataset,
            indices=indices,  # type: ignore
            chunksize=chunksize,
            shuffle_indices=shuffle_indices,
        )
        if verbose:
            LOGGER.info("Done.")

except ImportError as e:
    LOGGER.error(f"Error: {e}")
