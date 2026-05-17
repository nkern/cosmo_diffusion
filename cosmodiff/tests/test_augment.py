import pytest
import torch

from cosmodiff.augment import RandomDihedral2D, RandomFlip, RandomRot90, config_augmentations


def test_random_flip_uses_configured_dims():
    x = torch.arange(6).reshape(1, 2, 3)

    out = RandomFlip(dims=(-1,), p=1.0)(x)

    assert torch.equal(out, torch.flip(x, [-1]))


def test_random_flip_noop_when_no_dims_selected():
    x = torch.arange(6).reshape(1, 2, 3)

    out = RandomFlip(dims=(-1, -2), p=0.0)(x)

    assert torch.equal(out, x)


def test_random_rot90_noop_when_probability_is_zero():
    x = torch.arange(16).reshape(1, 4, 4)

    out = RandomRot90(dims=(-2, -1), p=0.0)(x)

    assert torch.equal(out, x)


def test_random_rot90_preserves_square_image_values():
    x = torch.arange(16).reshape(1, 4, 4)

    out = RandomRot90(dims=(-2, -1), p=1.0)(x)

    assert out.shape == x.shape
    assert torch.equal(out.flatten().sort().values, x.flatten().sort().values)


def test_random_dihedral_noop_when_probability_is_zero():
    x = torch.arange(16).reshape(1, 4, 4)

    out = RandomDihedral2D(dims=(-2, -1), p=0.0)(x)

    assert torch.equal(out, x)


def test_random_dihedral_preserves_square_image_values():
    x = torch.arange(16).reshape(1, 4, 4)

    out = RandomDihedral2D(dims=(-2, -1), p=1.0)(x)

    assert out.shape == x.shape
    assert torch.equal(out.flatten().sort().values, x.flatten().sort().values)


def test_square_symmetry_augmentations_require_two_dims():
    with pytest.raises(ValueError, match="requires exactly two dims"):
        RandomRot90(dims=(-1,))
    with pytest.raises(ValueError, match="requires exactly two dims"):
        RandomDihedral2D(dims=(-1,))


def test_config_augmentations_can_build_square_symmetry_pipeline():
    x = torch.arange(16).reshape(1, 4, 4)
    pipeline = config_augmentations({
        "RandomRot90": {"dims": [-2, -1], "p": 1.0},
        "RandomDihedral2D": {"dims": [-2, -1], "p": 1.0},
    })

    out = pipeline(x)

    assert out.shape == x.shape
    assert torch.equal(out.flatten().sort().values, x.flatten().sort().values)
