import os
import tempfile
import numpy as np
import torch
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler, DiTTransformer2DModel
from cosmodiff.utils import load_checkpoint, ArrayDataset
from cosmodiff.optim import train, generate, compute_fid, compute_kid, build_pca_encoder
from cosmodiff.augment import RandomRoll, RandomFlip


def test_train_basic():
    """Train a tiny UNet for 2 epochs, verify checkpoint saved and resumable."""
    model = UNet2DModel(
        sample_size=8,
        in_channels=1,
        out_channels=1,
        layers_per_block=1,
        block_out_channels=(16, 16),
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
        norm_num_groups=8,
    )

    images = torch.randn(16, 1, 8, 8)
    augmentations = torch.nn.Sequential(RandomRoll(dims=(-1,-2)), RandomFlip(dims=(-1, -2)))
    dataset = ArrayDataset(images, augmentations=augmentations)

    with tempfile.TemporaryDirectory() as tmp_dir:
        initial_weights = model.conv_in.weight.data.clone()
        metrics = train(
            dataset,
            model,
            noise_scheduler=DDPMScheduler(num_train_timesteps=10),
            num_epochs=2,
            batch_size=4,
            checkpoint_every_n_epochs=2,
            mixed_precision="no",
            output_dir=tmp_dir,
            force_cpu=True,
            verbose=False,
        )

        ckpt_path = os.path.join(tmp_dir, "checkpoint-epoch-0001")
        assert os.path.isdir(ckpt_path)
        assert os.path.exists(os.path.join(ckpt_path, "config.json"))
        assert os.path.exists(os.path.join(ckpt_path, "scheduler_config.json"))
        assert os.path.exists(os.path.join(ckpt_path, "checkpoint_config.yaml"))
        assert os.path.exists(os.path.join(ckpt_path, "augmentations.pkl"))
        assert os.path.exists(os.path.join(ckpt_path, "metrics.json"))

        # training checks: finite output, and weights changed
        assert all(torch.isfinite(torch.tensor(v)) for v in metrics["loss"])
        assert all(torch.isfinite(torch.tensor(v)) for v in metrics["epoch_loss"])
        assert not torch.allclose(model.conv_in.weight.data, initial_weights)

        # load_checkpoint
        _model, _noise_scheduler, _optimizer, _lr_scheduler, _augmentations = (
            load_checkpoint(ckpt_path)
        )

        assert _model is not None
        assert _noise_scheduler is not None
        assert _optimizer is not None
        assert _lr_scheduler is not None
        assert _augmentations is not None

        # continue training
        initial_weights = model.conv_in.weight.data.clone()
        metrics = train(
            dataset,
            model,
            noise_scheduler=DDPMScheduler(num_train_timesteps=10),
            num_epochs=2,
            batch_size=4,
            checkpoint_every_n_epochs=2,
            mixed_precision="no",
            output_dir=tmp_dir,
            force_cpu=True,
            verbose=False,
        )

        # training checks: finite output, and weights changed
        assert all(torch.isfinite(torch.tensor(v)) for v in metrics["loss"])
        assert all(torch.isfinite(torch.tensor(v)) for v in metrics["epoch_loss"])
        assert not torch.allclose(model.conv_in.weight.data, initial_weights)


def test_train_conditional_dit():
    """Train a tiny conditional DiT for 2 epochs."""
    model = DiTTransformer2DModel(
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=1,
        out_channels=1,
        num_layers=1,
        sample_size=4,
        patch_size=2,
        num_embeds_ada_norm=4,
        norm_num_groups=16,
    )

    images = torch.randn(16, 1, 4, 4)
    labels = torch.randint(0, 4, (16,))
    augmentations = torch.nn.Sequential(RandomRoll(dims=(-1,-2)), RandomFlip(dims=(-1, -2)))
    dataset = ArrayDataset(images, labels=labels, augmentations=augmentations)

    with tempfile.TemporaryDirectory() as tmp_dir:
        initial_weights = model.transformer_blocks[0].ff.net[0].proj.weight.clone()
        metrics = train(
            dataset,
            model,
            noise_scheduler=DDPMScheduler(num_train_timesteps=10),
            num_epochs=2,
            batch_size=4,
            checkpoint_every_n_epochs=2,
            mixed_precision="no",
            output_dir=tmp_dir,
            force_cpu=True,
            verbose=False,
        )

        ckpt_path = os.path.join(tmp_dir, "checkpoint-epoch-0001")
        assert os.path.isdir(ckpt_path)
        assert os.path.exists(os.path.join(ckpt_path, "config.json"))
        assert os.path.exists(os.path.join(ckpt_path, "scheduler_config.json"))
        assert os.path.exists(os.path.join(ckpt_path, "checkpoint_config.yaml"))
        assert os.path.exists(os.path.join(ckpt_path, "augmentations.pkl"))
        assert os.path.exists(os.path.join(ckpt_path, "metrics.json"))

        # training checks: finite output, and weights changed
        assert all(torch.isfinite(torch.tensor(v)) for v in metrics["loss"])
        assert all(torch.isfinite(torch.tensor(v)) for v in metrics["epoch_loss"])
        assert not torch.allclose(model.transformer_blocks[0].ff.net[0].proj.weight, initial_weights)

        # load_checkpoint
        _model, _noise_scheduler, _optimizer, _lr_scheduler, _augmentations = (
            load_checkpoint(ckpt_path)
        )

        assert _model is not None
        assert _noise_scheduler is not None
        assert _optimizer is not None
        assert _lr_scheduler is not None
        assert _augmentations is not None


def _make_unet(sample_size=8):
    return UNet2DModel(
        sample_size=sample_size,
        in_channels=1,
        out_channels=1,
        layers_per_block=1,
        block_out_channels=(16, 16),
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
        norm_num_groups=8,
    )


def test_generate_ddpm():
    """generate() returns the right shape and finite values with DDPMScheduler."""
    model = _make_unet()
    scheduler = DDPMScheduler(num_train_timesteps=10)
    images = generate(model, scheduler, batch_size=3, image_shape=(1, 8, 8))

    assert images.shape == (3, 1, 8, 8)
    assert torch.isfinite(images).all()


def test_generate_ddim_thinning():
    """ddim_thinning reduces inference steps and output is still valid."""
    model = _make_unet()
    scheduler = DDIMScheduler(num_train_timesteps=10)
    images = generate(model, scheduler, batch_size=2, image_shape=(1, 8, 8), ddim_thinning=2)

    assert images.shape == (2, 1, 8, 8)
    assert torch.isfinite(images).all()


def test_generate_renorm():
    """renorm callable is applied to the output."""
    model = _make_unet()
    scheduler = DDPMScheduler(num_train_timesteps=10)
    renorm = lambda x: (x + 1) / 2
    images = generate(model, scheduler, batch_size=2, image_shape=(1, 8, 8), renorm=renorm)

    assert images.shape == (2, 1, 8, 8)
    assert torch.isfinite(images).all()
    # renorm maps [-1,1] → [0,1]; generated values should shift accordingly
    assert images.mean() > 0


def test_generate_conditional_dit():
    """generate() works with a class-conditional DiT model and labels."""
    model = DiTTransformer2DModel(
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=1,
        out_channels=1,
        num_layers=1,
        sample_size=4,
        patch_size=2,
        num_embeds_ada_norm=4,
        norm_num_groups=16,
    )
    scheduler = DDPMScheduler(num_train_timesteps=10)
    labels = torch.tensor([0, 1, 2])
    images = generate(model, scheduler, batch_size=3, image_shape=(1, 4, 4), labels=labels)

    assert images.shape == (3, 1, 4, 4)
    assert torch.isfinite(images).all()


def test_generate_reproducible():
    """Same generator seed produces identical outputs."""
    model = _make_unet()
    scheduler = DDPMScheduler(num_train_timesteps=10)

    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    out1 = generate(model, scheduler, batch_size=2, image_shape=(1, 8, 8), generator=g1)
    out2 = generate(model, scheduler, batch_size=2, image_shape=(1, 8, 8), generator=g2)

    assert torch.allclose(out1, out2)


# ------------------------------------------------------------------ #
# Helpers for FID / KID tests                                         #
# ------------------------------------------------------------------ #


def _make_features(n: int, seed: int, mean: float = 0.0, std: float = 1.0):
    torch.manual_seed(seed)
    return (torch.randn(n, 1, 16, 16) * std + mean).double()


# ------------------------------------------------------------------ #
# FID tests                                                            #
# ------------------------------------------------------------------ #

def test_fid_finite_and_nonneg():
    """FID is finite and non-negative."""
    train_imgs = _make_features(500, seed=0)
    fake_imgs  = _make_features(500, seed=1)
    encode = build_pca_encoder(train_imgs, rank=16)
    fid = compute_fid(encode(train_imgs), encode(fake_imgs))
    assert torch.isfinite(torch.tensor(fid))
    assert fid >= 0.0


def test_fid_smaller_same_dist():
    """FID is smaller when fake comes from the same distribution vs a shifted one."""
    torch.manual_seed(0)
    train_imgs = torch.randn(500, 1, 16, 16).double()
    encode = build_pca_encoder(train_imgs, rank=16)
    feats_real = encode(train_imgs)

    torch.manual_seed(1)
    feats_same = encode(torch.randn(500, 1, 16, 16).double())

    torch.manual_seed(2)
    feats_diff = encode((torch.randn(500, 1, 16, 16) * 3 + 5).double())

    assert compute_fid(feats_real, feats_same) < compute_fid(feats_real, feats_diff)


# ------------------------------------------------------------------ #
# KID tests                                                            #
# ------------------------------------------------------------------ #

def test_kid_finite_and_nonneg():
    """KID mean is finite; std is non-negative."""
    train_imgs = _make_features(500, seed=0)
    fake_imgs  = _make_features(500, seed=1)
    encode = build_pca_encoder(train_imgs, rank=16)
    mean_kid, std_kid = compute_kid(encode(train_imgs), encode(fake_imgs), subset_size=200, n_subsets=5)
    assert torch.isfinite(torch.tensor(mean_kid))
    assert std_kid >= 0.0


def test_kid_smaller_same_dist():
    """KID mean is smaller when fake comes from the same distribution vs a shifted one."""
    torch.manual_seed(0)
    train_imgs = torch.randn(500, 1, 16, 16).double()
    encode = build_pca_encoder(train_imgs, rank=16)
    feats_real = encode(train_imgs)

    torch.manual_seed(1)
    feats_same = encode(torch.randn(500, 1, 16, 16).double())

    torch.manual_seed(2)
    feats_diff = encode((torch.randn(500, 1, 16, 16) * 3 + 5).double())

    kid_same, _ = compute_kid(feats_real, feats_same, subset_size=200, n_subsets=5)
    kid_diff, _ = compute_kid(feats_real, feats_diff, subset_size=200, n_subsets=5)
    assert kid_same < kid_diff



