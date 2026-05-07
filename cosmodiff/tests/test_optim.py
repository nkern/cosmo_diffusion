import os
import tempfile
import numpy as np
import torch
from diffusers import UNet2DModel, UNet2DConditionModel, DDPMScheduler, DDIMScheduler, DiTTransformer2DModel, PixArtTransformer2DModel
from cosmodiff.utils import load_checkpoint, ArrayDataset, find_latest_checkpoint
from cosmodiff.optim import train, generate, compute_fid, compute_kid, build_pca_encoder, synthesize_ema_from_checkpoints, compute_ema_profiles, load_ema_snapshot
from cosmodiff.augment import RandomRoll, RandomFlip
from cosmodiff.data import DATA_PATH

SIM_PATH = DATA_PATH / 'IllustrisTNG_Mcdm.npy'
PARAMS_PATH = DATA_PATH / 'params_Illustris.txt'
# shape: (34,) simulations × 6 cosmological parameters (a,b,c,d,e,f)
N_SIMS, N_PARAMS = 34, 6


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
    augmentations = torch.nn.Sequential(RandomRoll(size=8, dims=(-1,-2)), RandomFlip(dims=(-1, -2)))
    dataset = ArrayDataset(images, augmentations=augmentations)

    with tempfile.TemporaryDirectory() as tmp_dir:
        initial_weights = model.conv_in.weight.data.clone()
        result = train(
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
        assert all(torch.isfinite(torch.tensor(v)) for v in result['metrics']["loss"])
        assert all(torch.isfinite(torch.tensor(v)) for v in result['metrics']["epoch_loss"])
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

        # continue training from checkpoint
        initial_weights = model.conv_in.weight.data.clone()
        result = train(
            dataset,
            resume_from_checkpoint=ckpt_path,
            num_epochs=2,
            batch_size=4,
            checkpoint_every_n_epochs=2,
            mixed_precision="no",
            output_dir=tmp_dir,
            force_cpu=True,
            verbose=False,
        )

        # get new checkpoint: ensure it is epoch-0003
        ckpt_path2 = find_latest_checkpoint(tmp_dir)
        assert int(ckpt_path2.split('-')[-1]) == 3
        _model2, _noise_scheduler2, _optimizer2, _lr_scheduler2, _augmentations2 = (
            load_checkpoint(ckpt_path2)
        )

        # training checks: finite output, and weights changed
        assert all(torch.isfinite(torch.tensor(v)) for v in result['metrics']["loss"])
        assert all(torch.isfinite(torch.tensor(v)) for v in result['metrics']["epoch_loss"])
        assert not torch.allclose(_model2.conv_in.weight.data, initial_weights)


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
    augmentations = torch.nn.Sequential(RandomRoll(size=4, dims=(-1,-2)), RandomFlip(dims=(-1, -2)))
    dataset = ArrayDataset(images, labels=labels, augmentations=augmentations)

    with tempfile.TemporaryDirectory() as tmp_dir:
        initial_weights = model.transformer_blocks[0].ff.net[0].proj.weight.clone()
        result = train(
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
        assert all(torch.isfinite(torch.tensor(v)) for v in result['metrics']["loss"])
        assert all(torch.isfinite(torch.tensor(v)) for v in result['metrics']["epoch_loss"])
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


def _make_unet_class_cond(n_classes=4, sample_size=8):
    return UNet2DModel(
        sample_size=sample_size,
        in_channels=1,
        out_channels=1,
        layers_per_block=1,
        block_out_channels=(16, 16),
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
        norm_num_groups=8,
        num_class_embeds=n_classes + 1,  # last index is the null token
    )


def _make_unet_condition(sample_size=8, n_params=2):
    return UNet2DConditionModel(
        sample_size=sample_size,
        in_channels=1,
        out_channels=1,
        layers_per_block=1,
        block_out_channels=(16, 16),
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
        norm_num_groups=8,
        cross_attention_dim=16,
        encoder_hid_dim=n_params,
        encoder_hid_dim_type="text_proj",
    )


def _make_pixart(sample_size=8, n_params=N_PARAMS):
    return PixArtTransformer2DModel(
        sample_size=sample_size,
        patch_size=2,
        in_channels=1,
        out_channels=1,
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        cross_attention_dim=16,
        caption_channels=n_params,
        use_additional_conditions=False,
        norm_num_groups=None,
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


# ------------------------------------------------------------------ #
# CFG / conditioning tests                                             #
# ------------------------------------------------------------------ #

def test_train_cfg_class_labels():
    """train() with cfg_dropout runs without error and produces finite loss."""
    n_classes = 4
    model = _make_unet_class_cond(n_classes=n_classes)
    images = torch.randn(8, 1, 8, 8)
    labels = torch.randint(0, n_classes, (8,))
    dataset = ArrayDataset(images, labels=labels)

    with tempfile.TemporaryDirectory() as tmp_dir:
        result = train(
            dataset, model,
            noise_scheduler=DDPMScheduler(num_train_timesteps=10),
            num_epochs=1,
            batch_size=4,
            checkpoint_every_n_epochs=2,
            mixed_precision="no",
            output_dir=tmp_dir,
            force_cpu=True,
            verbose=False,
            cfg_dropout=0.5,
            conditioning='discrete',
        )
    assert all(torch.isfinite(torch.tensor(v)) for v in result['metrics']["loss"])


def test_train_encoder_hidden_states():
    """train() with conditioning='continuous' using real cosmological params."""
    params = np.loadtxt(PARAMS_PATH, dtype=np.float32)  # (34, 6)
    images = torch.randn(len(params), 1, 8, 8)
    labels = torch.as_tensor(params)  # (34, 6)
    dataset = ArrayDataset(images, labels=labels)
    model = _make_unet_condition(n_params=N_PARAMS)

    with tempfile.TemporaryDirectory() as tmp_dir:
        result = train(
            dataset, model,
            noise_scheduler=DDPMScheduler(num_train_timesteps=10),
            num_epochs=1,
            batch_size=4,
            checkpoint_every_n_epochs=2,
            mixed_precision="no",
            output_dir=tmp_dir,
            force_cpu=True,
            verbose=False,
            conditioning='continuous',
        )
    assert all(torch.isfinite(torch.tensor(v)) for v in result['metrics']["loss"])


def test_generate_encoder_hidden_states():
    """generate() with conditioning='continuous' using real cosmological params."""
    params = np.loadtxt(PARAMS_PATH, dtype=np.float32)  # (34, 6)
    model = _make_unet_condition(n_params=N_PARAMS)
    scheduler = DDPMScheduler(num_train_timesteps=10)
    # use first 3 simulations' parameters as conditioning
    labels = torch.as_tensor(params[:3])  # (3, 6)

    images = generate(
        model, scheduler,
        batch_size=3, image_shape=(1, 8, 8),
        labels=labels,
        conditioning='continuous',
    )
    assert images.shape == (3, 1, 8, 8)
    assert torch.isfinite(images).all()


def test_generate_cfg_guidance_scale():
    """guidance_scale != 1.0 produces different output than guidance_scale=1.0."""
    n_classes = 4
    model = _make_unet_class_cond(n_classes=n_classes)
    scheduler = DDPMScheduler(num_train_timesteps=10)
    labels = torch.tensor([0, 1, 2])

    g1 = torch.Generator().manual_seed(0)
    out_no_cfg = generate(
        model, scheduler,
        batch_size=3, image_shape=(1, 8, 8),
        labels=labels, guidance_scale=None,
        conditioning='discrete',
        generator=g1,
    )

    g2 = torch.Generator().manual_seed(0)
    out_cfg = generate(
        model, scheduler,
        batch_size=3, image_shape=(1, 8, 8),
        labels=labels, guidance_scale=2.0,
        conditioning='discrete',
        generator=g2,
    )

    assert out_no_cfg.shape == out_cfg.shape == (3, 1, 8, 8)
    assert torch.isfinite(out_cfg).all()
    assert not torch.allclose(out_no_cfg, out_cfg)


def test_train_pixart():
    """train() with PixArtTransformer2DModel and real cosmological params."""
    params = np.loadtxt(PARAMS_PATH, dtype=np.float32)  # (34, 6)
    images = torch.randn(len(params), 1, 8, 8)
    labels = torch.as_tensor(params)
    dataset = ArrayDataset(images, labels=labels)
    model = _make_pixart()

    with tempfile.TemporaryDirectory() as tmp_dir:
        result = train(
            dataset, model,
            noise_scheduler=DDPMScheduler(num_train_timesteps=10),
            num_epochs=1,
            batch_size=4,
            checkpoint_every_n_epochs=2,
            mixed_precision="no",
            output_dir=tmp_dir,
            force_cpu=True,
            verbose=False,
            conditioning='continuous',
        )
    assert all(torch.isfinite(torch.tensor(v)) for v in result['metrics']["loss"])


def test_generate_pixart():
    """generate() with PixArtTransformer2DModel and real cosmological params."""
    params = np.loadtxt(PARAMS_PATH, dtype=np.float32)
    model = _make_pixart()
    scheduler = DDPMScheduler(num_train_timesteps=10)
    labels = torch.as_tensor(params[:3])  # (3, 6)

    images = generate(
        model, scheduler,
        batch_size=3, image_shape=(1, 8, 8),
        labels=labels,
        conditioning='continuous',
    )
    assert images.shape == (3, 1, 8, 8)
    assert torch.isfinite(images).all()


# ------------------------------------------------------------------ #
# EMA tests                                                            #
# ------------------------------------------------------------------ #

def test_train_ema():
    """train() with ema_sigma_rels checkpoints EMA profiles and returns ema object."""
    model = _make_unet()
    images = torch.randn(8, 1, 8, 8)
    dataset = ArrayDataset(images)

    with tempfile.TemporaryDirectory() as tmp_dir:
        result = train(
            dataset, model,
            noise_scheduler=DDPMScheduler(num_train_timesteps=10),
            num_epochs=2,
            batch_size=4,
            checkpoint_every_n_epochs=2,
            mixed_precision="no",
            output_dir=tmp_dir,
            force_cpu=True,
            verbose=False,
            ema_sigma_rels=(0.05, 0.28),
        )

        assert result['ema'] is not None
        assert all(torch.isfinite(torch.tensor(v)) for v in result['metrics']["loss"])

        # EMA profiles should be checkpointed inside the epoch checkpoint dir
        ckpt_path = os.path.join(tmp_dir, "checkpoint-epoch-0001")
        assert os.path.isdir(os.path.join(ckpt_path, 'ema'))

        # should be able to synthesize a new EMA profile post-hoc
        synthesized = result['ema'].synthesize_ema_model(sigma_rel=0.15)
        assert synthesized is not None


def test_train_ema_multi_checkpoint_synthesis():
    """Multi-checkpoint pooling reduces synthesis residual and produces distinct models.

    Tests two properties from Karras et al. 2024:
    1. More checkpoints → smaller ||w_synth - w_target|| (better approximation).
    2. Different sigma_rel targets → different synthesized model weights.
    """
    import glob

    sigma_rels_train = (0.05, 0.28)
    sigma_rel_target = 0.15
    num_epochs = 9
    checkpoint_every = 3
    checkpoint_epochs = list(range(checkpoint_every - 1, num_epochs, checkpoint_every))

    model = _make_unet()
    dataset = ArrayDataset(torch.randn(8, 1, 8, 8))

    with tempfile.TemporaryDirectory() as tmp_dir:
        train(
            dataset, model,
            noise_scheduler=DDPMScheduler(num_train_timesteps=10),
            num_epochs=num_epochs,
            batch_size=4,
            checkpoint_every_n_epochs=checkpoint_every,
            mixed_precision="no",
            output_dir=tmp_dir,
            force_cpu=True,
            verbose=False,
            ema_sigma_rels=sigma_rels_train,
        )

        assert len(glob.glob(os.path.join(tmp_dir, 'checkpoint-epoch-*'))) == 3

        # 1. more checkpoints → smaller synthesis residual
        profiles_one = compute_ema_profiles(
            sigma_rels_train, checkpoint_epochs[:1], num_epochs, sigma_rel_target
        )
        profiles_all = compute_ema_profiles(
            sigma_rels_train, checkpoint_epochs, num_epochs, sigma_rel_target
        )
        residual_one = np.sum((profiles_one['synthesized'] - profiles_one['target']) ** 2)
        residual_all = np.sum((profiles_all['synthesized'] - profiles_all['target']) ** 2)
        assert residual_all < residual_one, \
            f"expected residual to decrease with more checkpoints: {residual_one:.6f} → {residual_all:.6f}"

        # 2. different sigma_rel targets → different synthesized model weights
        synth_lo = synthesize_ema_from_checkpoints(model, tmp_dir, sigma_rel_target=0.05)
        synth_hi = synthesize_ema_from_checkpoints(model, tmp_dir, sigma_rel_target=0.28)
        params_lo = torch.cat([p.flatten() for p in synth_lo.parameters()])
        params_hi = torch.cat([p.flatten() for p in synth_hi.parameters()])
        assert not torch.equal(params_lo, params_hi), \
            "sigma_rel=0.05 and sigma_rel=0.28 produced identical model weights"


def test_load_ema_snapshot():
    """load_ema_snapshot loads raw profile weights from a single checkpoint."""
    import glob
    model = _make_unet()
    dataset = ArrayDataset(torch.randn(8, 1, 8, 8))

    with tempfile.TemporaryDirectory() as tmp_dir:
        train(
            dataset, model,
            noise_scheduler=DDPMScheduler(num_train_timesteps=10),
            num_epochs=4, batch_size=4, checkpoint_every_n_epochs=2,
            mixed_precision="no", output_dir=tmp_dir, force_cpu=True, verbose=False,
            ema_sigma_rels=(0.05, 0.28),
        )

        ckpt_dirs = sorted(glob.glob(os.path.join(tmp_dir, 'checkpoint-epoch-*')))
        assert len(ckpt_dirs) == 2
        latest_ckpt = ckpt_dirs[-1]

        # different profiles should produce different weights
        m0 = load_ema_snapshot(_make_unet(), latest_ckpt, profile_index=0)
        m1 = load_ema_snapshot(_make_unet(), latest_ckpt, profile_index=1)
        p0 = torch.cat([p.flatten() for p in m0.parameters()])
        p1 = torch.cat([p.flatten() for p in m1.parameters()])
        assert not torch.equal(p0, p1), "profile 0 and 1 snapshots are identical"
        assert all(torch.isfinite(p).all() for p in m0.parameters())


def test_train_sigma_log_normal_sampling():
    """train() with sigma_log_normal samples timesteps and produces finite losses."""
    model = _make_unet()
    dataset = ArrayDataset(torch.randn(8, 1, 8, 8))
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = train(
            dataset, model,
            noise_scheduler=DDPMScheduler(num_train_timesteps=10, prediction_type='v_prediction'),
            num_epochs=2, batch_size=4,
            mixed_precision="no", output_dir=tmp_dir, force_cpu=True, verbose=False,
            sigma_log_normal=(-1.2, 1.2),
        )
        assert all(torch.isfinite(torch.tensor(v)) for v in result['metrics']['loss'])


def test_train_min_snr_weighting():
    """train() with min_snr_gamma set runs and produces finite losses for all 3 prediction types."""
    for prediction_type in ('epsilon', 'v_prediction', 'sample'):
        model = _make_unet()
        dataset = ArrayDataset(torch.randn(8, 1, 8, 8))
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = train(
                dataset, model,
                noise_scheduler=DDPMScheduler(num_train_timesteps=10, prediction_type=prediction_type),
                num_epochs=2, batch_size=4,
                mixed_precision="no", output_dir=tmp_dir, force_cpu=True, verbose=False,
                min_snr_gamma=5.0,
            )
            assert all(torch.isfinite(torch.tensor(v)) for v in result['metrics']['loss']), \
                f"non-finite loss with min_snr_gamma + prediction_type={prediction_type}"


def test_train_script():
    """cosmodiff_train.py main() runs end-to-end and writes a metrics file."""
    import sys
    import yaml
    from scripts.cosmodiff_train import main

    minimal_config = {
        "global": {"device": "cpu", "dtype": "float32"},
        "io": {"output_dir": None},  # filled in below
        "data": {
            "img_path": str(SIM_PATH),
            "img_read_fn": "npy_read_fn",
            "two_dim": True,
            "zthin": 4,
            "n_samples": 8,
            "keep_on_cpu": True,
            "log": False,
            "normalization": "center-max",
            "norm_kwargs": {"center": None, "xmax": None, "alpha": None, "beta": None},
        },
        "augmentations": {},
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
        "noise_scheduler": {
            "class": "DDPMScheduler",
            "kwargs": {"num_train_timesteps": 10},
        },
        "optimizer": {"class": "AdamW", "kwargs": {"lr": 1e-4}},
        "lr_scheduler": {"class": "ConstantLR", "kwargs": {"factor": 1.0, "total_iters": 0}},
        "train": {
            "num_epochs": 2,
            "batch_size": 4,
            "mixed_precision": "no",
            "checkpoint_every_n_epochs": 2,
            "force_cpu": True,
            "verbose": False,
        },
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        minimal_config["io"]["output_dir"] = tmp_dir
        config_path = os.path.join(tmp_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(minimal_config, f)

        orig_argv = sys.argv
        try:
            sys.argv = ["cosmodiff_train.py", "--config", config_path]
            main()
        finally:
            sys.argv = orig_argv

        metrics_files = [
            f for f in os.listdir(tmp_dir) if f.startswith("metrics_epoch_")
        ]
        assert len(metrics_files) == 1, f"expected 1 metrics file, got {metrics_files}"


def test_sample_script():
    """cosmodiff_sample.py main() runs end-to-end and writes an .npy file."""
    import sys
    from scripts.cosmodiff_sample import main

    # First train a tiny model so we have a real checkpoint on disk.
    model = _make_unet()
    dataset = ArrayDataset(torch.randn(8, 1, 8, 8))

    with tempfile.TemporaryDirectory() as tmp_dir:
        train(
            dataset, model,
            noise_scheduler=DDPMScheduler(num_train_timesteps=10),
            num_epochs=2, batch_size=4, checkpoint_every_n_epochs=2,
            mixed_precision="no", output_dir=tmp_dir, force_cpu=True, verbose=False,
        )

        out_npy = os.path.join(tmp_dir, "samples.npy")
        orig_argv = sys.argv
        try:
            sys.argv = [
                "cosmodiff_sample.py",
                "--output_dir", tmp_dir,
                "--n_samples", "4",
                "--batch_size", "2",
                "--image_shape", "1", "8", "8",
                "--ddim_thinning", "2",  # 5 inference steps for speed
                "--device", "cpu",
                "--output", out_npy,
                "--seed", "0",
            ]
            main()
        finally:
            sys.argv = orig_argv

        assert os.path.exists(out_npy), f"expected samples at {out_npy}"
        arr = np.load(out_npy)
        assert arr.shape == (4, 1, 8, 8), f"unexpected shape {arr.shape}"
        assert np.isfinite(arr).all(), "non-finite values in generated samples"


def test_sample_script_with_ema():
    """cosmodiff_sample.py with --ema_sigma_rel synthesizes EMA before sampling."""
    import sys
    from scripts.cosmodiff_sample import main

    model = _make_unet()
    dataset = ArrayDataset(torch.randn(8, 1, 8, 8))

    with tempfile.TemporaryDirectory() as tmp_dir:
        train(
            dataset, model,
            noise_scheduler=DDPMScheduler(num_train_timesteps=10),
            num_epochs=4, batch_size=4, checkpoint_every_n_epochs=2,
            mixed_precision="no", output_dir=tmp_dir, force_cpu=True, verbose=False,
            ema_sigma_rels=(0.05, 0.28),
        )

        out_npy = os.path.join(tmp_dir, "samples_ema.npy")
        orig_argv = sys.argv
        try:
            sys.argv = [
                "cosmodiff_sample.py",
                "--output_dir", tmp_dir,
                "--n_samples", "2",
                "--image_shape", "1", "8", "8",
                "--ddim_thinning", "5",
                "--device", "cpu",
                "--output", out_npy,
                "--ema_sigma_rel", "0.05",
                "--seed", "0",
            ]
            main()
        finally:
            sys.argv = orig_argv

        assert os.path.exists(out_npy)
        arr = np.load(out_npy)
        assert arr.shape == (2, 1, 8, 8)
        assert np.isfinite(arr).all()
