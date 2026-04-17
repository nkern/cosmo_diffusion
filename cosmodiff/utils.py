import torch
from torch.utils.data import Dataset


class ArrayDataset(Dataset):
    """Tensor dataset with optional augmentation applied at retrieval time."""

    def __init__(self, arrays: torch.Tensor, augment: callable = None):
        self.arrays = arrays
        self.augment = augment

    def __getitem__(self, idx):
        sample = self.arrays[idx]
        if self.augment is not None:
            sample = self.augment(sample)
        return sample
