import warnings
import math
from typing import List, Optional, Sequence, Tuple, TypeVar, Union, Dict
from torch import default_generator, randperm
from torch._utils import _accumulate
from torch import Generator
from torch.utils.data import Dataset

T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")
T_dict = Dict[str, T_co]
T_tuple = Tuple[T_co, ...]
T_stack = TypeVar("T_stack", T_tuple, T_dict)


class SplitSubset(Dataset[T_co]):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    dataset: Dataset[T_co]
    indices: Sequence[int]
    split: str

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int], split: str) -> None:
        self.dataset = dataset
        self.indices = indices
        self.split = split
        assert hasattr(dataset, "split"), "Dataset must have a split attribute"

    def __getitem__(self, idx):
        tmp = getattr(self.dataset, "split", None)

        setattr(self.dataset, "split", self.split)

        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]

        out = self.dataset[self.indices[idx]]

        setattr(self.dataset, "split", tmp)

        return out

    def __getitems__(self, indices: List[int]) -> List[T_co]:
        # add batched sampling support when parent dataset supports it.
        # see torch.utils.data._utils.fetch._MapDatasetFetcher

        tmp = getattr(self.dataset, "split", None)
        setattr(self.dataset, "split", self.split)
        if callable(getattr(self.dataset, "__getitems__", None)):
            out = self.dataset.__getitems__([self.indices[idx] for idx in indices])  # type: ignore[attr-defined]
        else:
            out = [self.dataset[self.indices[idx]] for idx in indices]

        setattr(self.dataset, "split", tmp)

        return out

    def __len__(self):
        return len(self.indices)


def named_random_split(
    dataset: Dataset[T],
    lengths: Sequence[Union[int, float]],
    split_names: Sequence[str],
    generator: Optional[Generator] = default_generator,
) -> List[SplitSubset[T]]:
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    Example:
        >>> # xdoctest: +SKIP
        >>> generator1 = torch.Generator().manual_seed(42)
        >>> generator2 = torch.Generator().manual_seed(42)
        >>> random_split(range(10), [3, 7], generator=generator1)
        >>> random_split(range(30), [0.3, 0.3, 0.4], generator=generator2)

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """

    assert len(lengths) == len(split_names), "Length of split_names must match lengths"

    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. " f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[arg-type, call-overload]

    return [
        SplitSubset(dataset, indices[offset - length : offset], split)
        for offset, length, split in zip(_accumulate(lengths), lengths, split_names)
    ]
