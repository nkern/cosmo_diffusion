import json
import tempfile
import numpy as np
import torch
from cosmodiff.utils import (
    ArrayDataset,
    load_data,
    minmax_norm,
    center_scale_norm,
    npy_read_fn,
    parse_config_model,
    parse_config_data,
    write_metrics,
    read_metrics,
)


def _make_array(n=20, nz=4, nx=8, ny=8):
    return np.random.rand(n, nz, nx, ny).astype(np.float32)


# ---------------------------------------------------------------------------
# load_data: n_samples / rng / memmap
# ---------------------------------------------------------------------------

def test_n_samples():
    arr = _make_array(n=20)
    images, _ = load_data(arr, img_read_fn=None, minmax=False, two_dim=False)
    assert images.shape[0] == 20

    images, _ = load_data(arr, img_read_fn=None, n_samples=7, minmax=False, two_dim=False)
    assert images.shape[0] == 7

    arr_t = torch.as_tensor(arr)
    for i in range(len(images)):
        assert any(torch.allclose(images[i], arr_t[j]) for j in range(len(arr_t)))


def test_rng():
    arr = _make_array(n=50)
    imgs1, _ = load_data(arr, img_read_fn=None, n_samples=10, rng=np.random.default_rng(42), minmax=False, two_dim=False)
    imgs2, _ = load_data(arr, img_read_fn=None, n_samples=10, rng=np.random.default_rng(42), minmax=False, two_dim=False)
    imgs3, _ = load_data(arr, img_read_fn=None, n_samples=10, rng=np.random.default_rng(99), minmax=False, two_dim=False)
    assert torch.allclose(imgs1, imgs2)
    assert not torch.allclose(imgs1, imgs3)


def test_memmap():
    arr = _make_array(n=10)
    with tempfile.NamedTemporaryFile(suffix=".npy") as f:
        np.save(f.name, arr)
        mmap = np.load(f.name, mmap_mode="r")
        images_all, _ = load_data(mmap, img_read_fn=None, minmax=False, two_dim=False)
        images_sub, _ = load_data(mmap, img_read_fn=None, n_samples=5, minmax=False, two_dim=False)
    assert images_all.shape[0] == 10
    assert images_sub.shape[0] == 5


# ---------------------------------------------------------------------------
# ArrayDataset
# ---------------------------------------------------------------------------

def test_array_dataset():
    images = torch.randn(10, 1, 8, 8)
    labels = torch.arange(10)

    ds = ArrayDataset(images)
    assert len(ds) == 10
    assert set(ds[0].keys()) == {"images"}

    ds_labeled = ArrayDataset(images, labels=labels)
    item = ds_labeled[3]
    assert item["labels"] == 3
    assert torch.allclose(item["images"], images[3])


# ---------------------------------------------------------------------------
# minmax_norm / center_scale_norm
# ---------------------------------------------------------------------------

def test_minmax_norm():
    x = torch.randn(4, 8, 8)
    out = minmax_norm(x)
    assert out.min().item() >= -1.0 - 1e-6
    assert out.max().item() <= 1.0 + 1e-6


def test_center_scale_norm():
    x = torch.randn(100)
    out, avg, std = center_scale_norm(x.clone(), scale=10)
    assert abs(out.median().item()) < 0.1
    assert out.abs().max().item() < 2.0


# ---------------------------------------------------------------------------
# npy_read_fn
# ---------------------------------------------------------------------------

def test_npy_read_fn():
    arr = np.random.rand(5, 4, 8, 8).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".npy") as f:
        np.save(f.name, arr)
        result = npy_read_fn(f.name)
    assert result.shape == arr.shape
    assert np.allclose(result, arr)


# ---------------------------------------------------------------------------
# write_metrics / read_metrics
# ---------------------------------------------------------------------------

def test_write_read_metrics():
    metrics = {"loss": [1.0, 0.8], "epoch_loss": [0.9], "epoch_lr": [1e-4], "epoch_times": [10.0]}
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        write_metrics(metrics, f.name)
        recovered = read_metrics(f.name)
    assert recovered == metrics


# ---------------------------------------------------------------------------
# parse_config_model
# ---------------------------------------------------------------------------

def test_parse_config_model():
    config = {
        "global": {"device": "cpu"},
        "model": {
            "class": "UNet2DModel",
            "kwargs": {
                "sample_size": 8,
                "in_channels": 1,
                "out_channels": 1,
                "layers_per_block": 1,
                "block_out_channels": [16, 16],
                "down_block_types": ["DownBlock2D", "DownBlock2D"],
                "up_block_types": ["UpBlock2D", "UpBlock2D"],
                "norm_num_groups": 8,
            },
        },
        "optimizer": {"class": "AdamW", "kwargs": {"lr": 1e-4}},
        "noise_scheduler": {"class": "DDPMScheduler", "kwargs": {"num_train_timesteps": 10}},
        "lr_scheduler": {"class": "ConstantLR", "kwargs": {"factor": 1.0, "total_iters": 0}},
    }
    model, optimizer, noise_scheduler, lr_scheduler = parse_config_model(config)
    assert model is not None
    assert optimizer is not None
    assert noise_scheduler is not None
    assert lr_scheduler is not None


# ---------------------------------------------------------------------------
# parse_config_data
# ---------------------------------------------------------------------------

def test_parse_config_data():
    arr = _make_array(n=6, nz=4, nx=8, ny=8)
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f.name, arr)
        img_path = f.name

    config = {
        "global": {"device": "cpu", "dtype": "float32"},
        "data": {
            "img_path": img_path,
            "img_read_fn": "npy_read_fn",
            "log": False,
            "minmax": True,
            "two_dim": True,
            "zthin": 1,
            "keep_on_cpu": True,
        },
    }
    dataset = parse_config_data(config)
    assert len(dataset) == 6 * 4  # Nbatch * Nz slices
    item = dataset[0]
    assert "images" in item
    assert item["images"].shape == torch.Size([1, 8, 8])
