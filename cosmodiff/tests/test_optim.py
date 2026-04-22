import os
import tempfile
import numpy as np
import torch
from diffusers import UNet2DModel, DDPMScheduler, DiTTransformer2DModel
from cosmodiff.utils import load_checkpoint, ArrayDataset
from cosmodiff.optim import train
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

		ckpt_path = os.path.join(tmp_dir, "checkpoint-epoch-1")
		assert os.path.isdir(ckpt_path)
		assert os.path.exists(os.path.join(ckpt_path, "noise_scheduler.pkl"))
		assert os.path.exists(os.path.join(ckpt_path, "optimizer.pkl"))
		assert os.path.exists(os.path.join(ckpt_path, "lr_scheduler.pkl"))
		assert os.path.exists(os.path.join(ckpt_path, "augmentations.pkl"))

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

		ckpt_path = os.path.join(tmp_dir, "checkpoint-epoch-1")
		assert os.path.isdir(ckpt_path)
		assert os.path.exists(os.path.join(ckpt_path, "noise_scheduler.pkl"))
		assert os.path.exists(os.path.join(ckpt_path, "optimizer.pkl"))
		assert os.path.exists(os.path.join(ckpt_path, "lr_scheduler.pkl"))
		assert os.path.exists(os.path.join(ckpt_path, "augmentations.pkl"))

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




