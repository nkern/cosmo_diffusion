import copy
import json
import tempfile
import numpy as np
import torch
from cosmodiff.utils import (
    ArrayDataset,
    load_data,
    npy_read_fn,
    txt_read_fn,
    parse_config_model,
    parse_config_data,
    read_config,
    write_metrics,
    read_metrics,
)
from cosmodiff.transform import Normalization, MultiNormalization, Transform, MultiTransform
from cosmodiff.data import DATA_PATH

CONFIG_PATH = DATA_PATH / 'config.yaml'
SIM_PATH = DATA_PATH / 'IllustrisTNG_Mcdm.npy'
PARAMS_PATH = DATA_PATH / 'params_Illustris.txt'
# shape: (34, 32, 32, 32), dtype: float32
DATA_N, DATA_NZ, DATA_NX, DATA_NY = 34, 32, 32, 32


def _make_array(n=20, nz=4, nx=8, ny=8):
    return np.random.rand(n, nz, nx, ny).astype(np.float32)


# ---------------------------------------------------------------------------
# load_data: n_samples / seed / memmap
# ---------------------------------------------------------------------------

def test_n_samples():
    images_all = load_data(SIM_PATH, img_read_fn=npy_read_fn, normalization=None, reshape="3d")['images']
    assert images_all.shape[0] == DATA_N

    images_sub = load_data(SIM_PATH, img_read_fn=npy_read_fn, n_samples=3, normalization=None, reshape="3d")['images']
    assert images_sub.shape[0] == 3

    for i in range(len(images_sub)):
        assert any(torch.allclose(images_sub[i], images_all[j]) for j in range(len(images_all)))


def test_n_samples_labels_in_sync():
    """Labels must be subsampled with the same indices as images."""
    rng = np.random.default_rng(0)
    arr = rng.random((20, 4, 8, 8)).astype(np.float32)
    # labels encode each sample's original row index so we can verify alignment
    labels = np.arange(20)

    out = load_data(
        arr, img_read_fn=None,
        label_path=labels, label_read_fn=None,
        n_samples=7, seed=0,
        normalization=None, reshape="3d",
    )
    images, out_labels = out['images'], out['labels']
    assert images.shape[0] == 7
    assert out_labels.shape[0] == 7

    arr_t = torch.as_tensor(arr)
    for img, lbl in zip(images, out_labels):
        # the label is the original index; confirm the image matches that row
        assert torch.allclose(img, arr_t[lbl.item()])


def test_seed():
    imgs1 = load_data(SIM_PATH, img_read_fn=npy_read_fn, n_samples=3, seed=42, normalization=None, reshape="3d")['images']
    imgs2 = load_data(SIM_PATH, img_read_fn=npy_read_fn, n_samples=3, seed=42, normalization=None, reshape="3d")['images']
    imgs3 = load_data(SIM_PATH, img_read_fn=npy_read_fn, n_samples=3, seed=99, normalization=None, reshape="3d")['images']
    assert torch.allclose(imgs1, imgs2)
    assert not torch.allclose(imgs1, imgs3)


def test_memmap():
    mmap = np.load(SIM_PATH, mmap_mode="r")
    images_all = load_data(mmap, img_read_fn=None, normalization=None, reshape="3d")['images']
    images_sub = load_data(mmap, img_read_fn=None, n_samples=3, normalization=None, reshape="3d")['images']
    assert images_all.shape[0] == DATA_N
    assert images_sub.shape[0] == 3


# ---------------------------------------------------------------------------
# load_data: multi-path returns the right (Multi)Normalization / (Multi)Transform
# ---------------------------------------------------------------------------

def test_load_data_multipath_norm_transform_types():
    """Two img_paths + two label_paths exercising the two normalization modes.

    Mode A — single normalization / transform shared across all paths
        → ``out['norm']`` is :class:`Normalization`
        → ``out['tform']`` is :class:`Transform`

    Mode B — list-shaped normalization / transform (one per path)
        → ``out['norm']`` is :class:`MultiNormalization`
        → ``out['tform']`` is :class:`MultiTransform`
    """
    img_paths = [str(SIM_PATH), str(SIM_PATH)]
    label_paths = [str(PARAMS_PATH), str(PARAMS_PATH)]

    # --- Case (i): scalar normalization / transform ---
    out_a = load_data(
        img_path=img_paths,
        img_read_fn=npy_read_fn,
        label_path=label_paths,
        label_read_fn=txt_read_fn,
        reshape='2d',
        zthin=4,
        normalization='min-max',
        transform=['log'],
    )
    assert isinstance(out_a['norm'], Normalization)
    assert not isinstance(out_a['norm'], MultiNormalization)
    assert isinstance(out_a['tform'], Transform)
    assert not isinstance(out_a['tform'], MultiTransform)

    # --- Case (ii): per-path normalization / transform ---
    out_b = load_data(
        img_path=img_paths,
        img_read_fn=npy_read_fn,
        label_path=label_paths,
        label_read_fn=txt_read_fn,
        reshape='2d',
        zthin=4,
        normalization=['min-max', 'min-max'],
        norm_kwargs=[{}, {}],
        transform=[['log'], ['log']],
    )
    assert isinstance(out_b['norm'], MultiNormalization)
    assert isinstance(out_b['tform'], MultiTransform)


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
# npy_read_fn
# ---------------------------------------------------------------------------

def test_npy_read_fn():
    result = npy_read_fn(SIM_PATH)
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
    out = parse_config_model(config)
    assert out['model'] is not None
    assert out['optimizer'] is not None
    assert out['noise_scheduler'] is not None
    assert out['lr_scheduler'] is not None


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
            "normalization": "min-max",
            "reshape": "2d",
            "zthin": 1,
            "keep_on_cpu": True,
        },
    }
    out = parse_config_data(config)
    dataset = out['data']
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

    out = parse_config_model(config)
    model = out['model']
    optimizer = out['optimizer']
    noise_scheduler = out['noise_scheduler']
    lr_scheduler = out['lr_scheduler']

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
    cfg['data']['img_path'] = str(SIM_PATH)
    cfg['data']['label_path'] = None
    cfg['data']['keep_on_cpu'] = True
    cfg['data']['transform'] = None  # this test checks raw shape, not transform behavior

    out = parse_config_data(cfg)
    dataset = out['data']
    norm = out['norm']

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
