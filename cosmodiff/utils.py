import torch
from torch.utils.data import Dataset


class ArrayDataset(Dataset):
    """Tensor dataset with optional augmentation applied at retrieval time."""

    def __init__(self, *arrays: torch.Tensor, augment: callable = None):
        if not arrays:
            raise ValueError("At least one array is required.")
        length = arrays[0].shape[0]
        if not all(a.shape[0] == length for a in arrays):
            raise ValueError("All arrays must have the same length along dim 0.")
        self.arrays = arrays
        self.augment = augment

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, idx):
        sample = tuple(a[idx] for a in self.arrays)
        if self.augment is not None:
            sample = self.augment(sample)
        return sample
