import numpy as np
import torch
from torch import nn


class RandomRoll(nn.Module):
    """Randomly roll a tensor along specified dimensions.

    Args:
        dims (tuple of int): Dimensions to roll along. Defaults to ``(-1,)``.
    """
    def __init__(self, size=128, dims=(-1,)):
        super().__init__()
        self.size = size
        self.dims = tuple(dims)
        self.ndim = len(dims)

    def __call__(self, x):
        if x is None:
            return None
        shift = torch.randint(0, self.size, (self.ndim,), dtype=torch.long, device='cpu').tolist()
        return torch.roll(x, shift, self.dims)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dims={self.dims})"


class RandomCrop(nn.Module):
    """Randomly crop a tensor along specified dimensions.

    Args:
        size (int or tuple of int): Output size for each cropped dimension.
            If a single int, the same size is used for all dims.
        dims (tuple of int): Dimensions to crop along. Defaults to ``(-2, -1)``.
    """
    def __init__(self, size, dims=(-2, -1)):
        super().__init__()
        self.dims = tuple(dims)
        self.size = (size,) * len(dims) if isinstance(size, int) else tuple(size)
        if len(self.size) != len(self.dims):
            raise ValueError("size and dims must have the same length.")

    def __call__(self, x):
        if x is None:
            return None
        slices = [slice(None)] * x.ndim
        for dim, size in zip(self.dims, self.size):
            dim = dim % x.ndim  # normalise negative dims
            max_start = x.shape[dim] - size
            if max_start < 0:
                raise ValueError(
                    f"Crop size {size} is larger than tensor size {x.shape[dim]} along dim {dim}."
                )
            start = torch.randint(0, max_start + 1, (1,)).item()
            slices[dim] = slice(start, start + size)
        return x[tuple(slices)]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, dims={self.dims})"


class RandomFlip(nn.Module):
    """Randomly flip a tensor along specified dimensions.

    Args:
        dims (tuple of int): Dimensions eligible for flipping. Defaults to ``(-2, -1)``.
        p (float): Probability of flipping along each dim independently.
            Defaults to ``0.5``.
    """
    def __init__(self, dims=(-2, -1), p=0.5):
        super().__init__()
        self.dims = list(dims)
        self.p = p

    def __call__(self, x):
        if x is None:
            return None
        draw = torch.rand(len(self.dims), device='cpu')
        flip = torch.where(draw < self.p)[0].tolist()
        if len(flip) == 0:
            return x
        else:
            return torch.flip(x, [self.dims[f] for f in flip])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dims={self.dims}, p={self.p})"


class RandomMove(nn.Module):
    """Randomly swap along specified dimensions.

    Args:
        dims (tuple of int): Dimensions to randomly swap.
    """
    def __init__(self, dims=(-2, -1)):
        super().__init__()
        self.dims = tuple(dims)

    def __call__(self, x):
        if x is None:
            return None

        r = torch.randperm(len(self.dims))
        new_dims = torch.tensor(self.dims)[r].tolist()

        return torch.movedim(x, self.dims, new_dims)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dims={self.dims})"


def config_augmentations(augmentations):
    """Configure a chain of augmentations.

    Args:
        augmentations (dictionary): keys are str of class objects,
            values are kwargs for that class.
    """
    from cosmodiff import augment
    pipeline = []
    for name, kwargs in augmentations.items():
        pipeline.append(getattr(augment, name)(**kwargs))

    return nn.Sequential(*pipeline)

