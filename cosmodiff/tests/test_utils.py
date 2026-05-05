import copy
import importlib.resources
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from cosmodiff.utils import (
    ArrayDataset,
    load_data,
    minmax_norm,
    center_max_norm,
    tanh_norm,
    Normalization,
    npy_read_fn,
    parse_config_model,
    parse_config_data,
    read_config,
    write_metrics,
    read_metrics,
)

CONFIG_PATH = importlib.resources.files('cosmodiff.data') / 'config.yaml'
DATA_PATH = importlib.resources.files('cosmodiff.data') / 'IllustrisTNG_Mcdm.npy'
# shape: (34, 32, 32, 32), dtype: float32
DATA_N, DATA_NZ, DATA_NX, DATA_NY = 34, 32, 32, 32


def _make_array(n=20, nz=4, nx=8, ny=8):
    return np.random.rand(n, nz, nx, ny).astype(np.float32)


# ---------------------------------------------------------------------------
# load_data: n_samples / seed / memmap
# ---------------------------------------------------------------------------

def test_n_samples():
    images_all, _, _ = load_data(DATA_PATH, img_read_fn=npy_read_fn, normalization=None, two_dim=False)
    assert images_all.shape[0] == DATA_N

    images_sub, _, _ = load_data(DATA_PATH, img_read_fn=npy_read_fn, n_samples=3, normalization=None, two_dim=False)
    assert images_sub.shape[0] == 3

    for i in range(len(images_sub)):
        assert any(torch.allclose(images_sub[i], images_all[j]) for j in range(len(images_all)))


def test_n_samples_labels_in_sync():
    """Labels must be subsampled with the same indices as images."""
    rng = np.random.default_rng(0)
    arr = rng.random((20, 4, 8, 8)).astype(np.float32)
    # labels encode each sample's original row index so we can verify alignment
    labels = np.arange(20)

    images, out_labels, _ = load_data(
        arr, img_read_fn=None,
        label_path=labels, label_read_fn=None,
        n_samples=7, seed=0,
        normalization=None, two_dim=False,
    )
    assert images.shape[0] == 7
    assert out_labels.shape[0] == 7

    arr_t = torch.as_tensor(arr)
    for img, lbl in zip(images, out_labels):
        # the label is the original index; confirm the image matches that row
        assert torch.allclose(img, arr_t[lbl.item()])


def test_seed():
    imgs1, _, _ = load_data(DATA_PATH, img_read_fn=npy_read_fn, n_samples=3, seed=42, normalization=None, two_dim=False)
    imgs2, _, _ = load_data(DATA_PATH, img_read_fn=npy_read_fn, n_samples=3, seed=42, normalization=None, two_dim=False)
    imgs3, _, _ = load_data(DATA_PATH, img_read_fn=npy_read_fn, n_samples=3, seed=99, normalization=None, two_dim=False)
    assert torch.allclose(imgs1, imgs2)
    assert not torch.allclose(imgs1, imgs3)


def test_memmap():
    mmap = np.load(DATA_PATH, mmap_mode="r")
    images_all, _, _ = load_data(mmap, img_read_fn=None, normalization=None, two_dim=False)
    images_sub, _, _ = load_data(mmap, img_read_fn=None, n_samples=3, normalization=None, two_dim=False)
    assert images_all.shape[0] == DATA_N
    assert images_sub.shape[0] == 3


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
# minmax_norm / center_max_norm / tanh_norm
# ---------------------------------------------------------------------------

def test_minmax_norm():
    # use non-negative data so the formula x *= 2/xmax is well-defined
    x = torch.rand(4, 8, 8)
    out, params = minmax_norm(x)
    assert out.min().item() >= -1.0 - 1e-6
    assert out.max().item() <= 1.0 + 1e-6
    assert 'xmin' in params and 'xmax' in params


def test_minmax_norm_inverse():
    x = torch.rand(4, 8, 8)
    out, params = minmax_norm(x)
    recovered, _ = minmax_norm(out, inverse=True, **params)
    assert torch.allclose(recovered, x, atol=1e-5)


def test_minmax_norm_inplace():
    x = torch.rand(4, 8, 8)
    x_clone = x.clone()
    out, _ = minmax_norm(x, inplace=True)
    assert out.data_ptr() == x.data_ptr()
    _, params2 = minmax_norm(x_clone)
    assert torch.allclose(out, x_clone.sub(params2['xmin']).mul(2 / params2['xmax']).sub(1), atol=1e-5)


def test_center_max_norm():
    x = torch.randn(100)
    out, params = center_max_norm(x.clone())
    assert abs(out.mean().item()) < 0.1
    assert out.abs().max().item() <= 1.0 + 1e-6
    assert 'center' in params and 'xmax' in params


def test_center_max_norm_inverse():
    x = torch.randn(100)
    out, params = center_max_norm(x.clone())
    recovered, _ = center_max_norm(out, inverse=True, **params)
    assert torch.allclose(recovered, x, atol=1e-5)


def test_tanh_norm():
    x = torch.randn(100)
    out, params = tanh_norm(x)
    assert out.shape == x.shape
    assert set(params.keys()) == {'mu', 'alpha', 'beta', 'delta', 'gamma', 'sigma'}
    # tanh output is strictly bounded by sigma * (-beta, alpha)
    assert out.min().item() > -params['beta'] * params['sigma']
    assert out.max().item() < params['alpha'] * params['sigma']


def test_tanh_norm_sigma():
    x = torch.randn(100)
    out_default, _ = tanh_norm(x, sigma=1.0)
    out_scaled, params = tanh_norm(x, sigma=2.0)
    assert torch.allclose(out_scaled, out_default * 2.0, atol=1e-6)
    assert params['sigma'] == 2.0


def test_tanh_norm_inverse():
    x = torch.randn(100) * 0.5  # keep within tanh saturation limits
    out, params = tanh_norm(x)
    recovered, _ = tanh_norm(out, inverse=True, **params)
    assert torch.allclose(recovered, x, atol=1e-5)


# ---------------------------------------------------------------------------
# Normalization class
# ---------------------------------------------------------------------------

def test_normalization_minmax():
    x = torch.rand(10, 1, 8, 8)
    norm = Normalization('min-max', inplace=False)
    out = norm(x)
    assert out is not None
    assert out.shape == x.shape
    assert out.min().item() >= -1.0 - 1e-6
    assert out.max().item() <= 1.0 + 1e-6


def test_normalization_centermax():
    x = torch.randn(10, 1, 8, 8)
    norm = Normalization('center-max', inplace=False)
    out = norm(x)
    assert out is not None
    assert out.shape == x.shape
    assert out.abs().max().item() <= 1.0 + 1e-6


def test_normalization_inverse():
    x = torch.rand(10, 1, 8, 8)
    norm = Normalization('min-max', inplace=False)
    out = norm(x)
    recovered = norm.inverse(out)
    assert torch.allclose(recovered, x, atol=1e-5)


def test_normalization_tanh():
    # tanh branch: minmax first, then tanh — output is strictly within sigma*(-beta, alpha)
    x = torch.rand(10, 1, 8, 8)
    norm = Normalization('tanh', inplace=False)
    out = norm(x)
    assert out.shape == x.shape
    assert out.min().item() > -norm.kwargs['beta'] * norm.kwargs['sigma']
    assert out.max().item() < norm.kwargs['alpha'] * norm.kwargs['sigma']
    # params from both stages must be stored after forward
    assert all(k in norm.kwargs for k in ('center', 'xmax', 'mu', 'alpha', 'beta', 'gamma', 'delta', 'sigma'))


def test_normalization_tanh_inverse():
    x = torch.rand(10, 1, 8, 8)
    norm = Normalization('tanh', inplace=False)
    out = norm(x)
    recovered = norm.inverse(out)
    assert torch.allclose(recovered, x, atol=1e-5)


def test_normalization_params_fixed_when_given():
    """Params supplied at init must not be overwritten by the forward pass."""
    fixed_xmin = torch.tensor(0.0)
    fixed_xmax = torch.tensor(2.0)
    norm = Normalization('min-max', inplace=False, xmin=fixed_xmin, xmax=fixed_xmax)

    # forward on data whose natural min/max differ from the fixed values
    x = torch.rand(10, 1, 8, 8) * 10 + 5
    norm(x)

    assert torch.equal(norm.kwargs['xmin'], fixed_xmin), "xmin was overwritten"
    assert torch.equal(norm.kwargs['xmax'], fixed_xmax), "xmax was overwritten"


def test_normalization_params_inferred_on_first_forward():
    """Params not supplied at init must be populated after the first forward pass."""
    import copy
    norm = Normalization('min-max', inplace=False)
    assert 'xmin' not in norm.kwargs
    assert 'xmax' not in norm.kwargs

    x = torch.rand(10, 1, 8, 8)
    norm(x)

    assert 'xmin' in norm.kwargs, "xmin not set after first forward"
    assert 'xmax' in norm.kwargs, "xmax not set after first forward"
    assert np.allclose(norm.kwargs['xmin'], x.min().item())
    assert np.allclose(norm.kwargs['xmax'], x.max().item())

    # second forward on different data must reuse the inferred params, not recompute
    y = torch.rand(10, 1, 8, 8) * 10 + 5
    xmin_after_first = copy.copy(norm.kwargs['xmin'])
    xmax_after_first = copy.copy(norm.kwargs['xmax'])
    norm(y)

    assert np.allclose(norm.kwargs['xmin'], xmin_after_first), "xmin changed on second forward"
    assert np.allclose(norm.kwargs['xmax'], xmax_after_first), "xmax changed on second forward"


# ---------------------------------------------------------------------------
# npy_read_fn
# ---------------------------------------------------------------------------

def test_npy_read_fn():
    result = npy_read_fn(DATA_PATH)
    assert result.shape == (DATA_N, DATA_NZ, DATA_NX, DATA_NY)
    assert result.dtype == np.float32


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
            "norm": "min-max",
            "two_dim": True,
            "zthin": 1,
            "keep_on_cpu": True,
        },
    }
    dataset, norm = parse_config_data(config)
    assert len(dataset) == 6 * 4  # Nbatch * Nz slices
    item = dataset[0]
    assert "images" in item
    assert item["images"].shape == torch.Size([1, 8, 8])


# ---------------------------------------------------------------------------
# configs/config.yaml round-trip
# ---------------------------------------------------------------------------

def test_config_yaml_parse_model():
    """parse_config_model must instantiate all four components from config.yaml."""
    config = read_config(CONFIG_PATH)
    config['global']['device'] = 'cpu'  # don't require GPU in CI

    model, optimizer, noise_scheduler, lr_scheduler = parse_config_model(config)

    assert type(model).__name__ == 'UNet2DModel'
    assert model.config.sample_size == 64
    assert model.config.in_channels == 1
    assert model.config.out_channels == 1

    assert type(optimizer).__name__ == 'AdamW'
    assert abs(optimizer.param_groups[0]['lr'] - 1e-4) < 1e-10
    assert abs(optimizer.param_groups[0]['weight_decay'] - 1e-2) < 1e-10

    assert type(noise_scheduler).__name__ == 'DDPMScheduler'
    assert noise_scheduler.config.num_train_timesteps == 1000

    assert type(lr_scheduler).__name__ == 'ConstantLR'


def test_config_yaml_parse_data():
    """parse_config_data must build an ArrayDataset with the right shape,
    normalization, and augmentation pipeline from config.yaml."""
    from cosmodiff.augment import RandomRoll, RandomFlip

    config = read_config(CONFIG_PATH)
    config['global']['device'] = 'cpu'

    cfg = copy.deepcopy(config)
    cfg['data']['img_path'] = str(DATA_PATH)
    cfg['data']['label_path'] = None
    cfg['data']['keep_on_cpu'] = True

    dataset, norm = parse_config_data(cfg)

    zthin = config['data']['zthin']
    assert isinstance(dataset, ArrayDataset)
    assert len(dataset) == DATA_N * (DATA_NZ // zthin)
    item = dataset[0]
    assert 'images' in item
    assert item['images'].shape == torch.Size([1, DATA_NX, DATA_NY])

    assert isinstance(norm, Normalization)
    assert norm.method == config['data']['normalization']

    assert dataset.augmentations is not None
    aug_types = [type(t) for t in dataset.augmentations]
    assert RandomRoll in aug_types
    assert RandomFlip in aug_types
