import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ConstantLR
from accelerate import Accelerator
from diffusers import AutoModel, DDPMScheduler
from tqdm.auto import tqdm
import time


class ArrayDataset(Dataset):
	"""Tensor dataset with optional augmentation and class labels, applied at
	retrieval time.

	Args:
		arrays (torch.Tensor): Image tensor of shape ``(N, C, [D], H, W)``.
		labels (torch.Tensor, optional): LongTensor of shape ``(N,)``. When
			provided, ``__getitem__`` returns a dict with ``"images"`` and
			``"labels"`` keys. When ``None``, returns a dict with an
			``"images"`` key only.
		augmentations (callable, optional): Augmentation pipeline applied to
			each sample at retrieval time. Defaults to ``None``.

	Example:
		Unconditional::

			dataset = ArrayDataset(images)
			# {"images": tensor}

		Class-conditional::

			dataset = ArrayDataset(images, labels=labels)
			# {"images": tensor, "labels": tensor}
	"""
	def __init__(self, arrays: torch.Tensor, labels: torch.Tensor = None, augmentations: callable = None):
		self.arrays = arrays
		self.labels = labels
		self.augmentations = augmentations

	def __len__(self):
		return len(self.arrays)

	def __getitem__(self, idx):
		sample = self.arrays[idx]
		if self.augmentations is not None:
			sample = self.augmentations(sample)
		if self.labels is not None:
			return {"images": sample, "labels": self.labels[idx]}
		return {"images": sample}


def load_checkpoint(ckpt_path: str):
	"""Reconstruct model, noise_scheduler, optimizer, lr_scheduler, and
	optionally an augmentation pipeline from a saved checkpoint directory
	produced by ``train()``.

	All objects are returned fully reconstructed and ready to be passed
	directly into ``train()``. The lr_scheduler and noise_scheduler are
	stored as pickles so no knowledge of the original class is required.
	The augmentation pipeline is restored from ``augmentations.pt`` if
	present in the checkpoint directory and should be re-attached to the
	dataset before passing it to ``train()``.

	Args:
		ckpt_path (str): Path to a checkpoint directory produced by ``train()``.

	Returns:
		tuple: ``(model, noise_scheduler, optimizer, lr_scheduler, augmentations)``
		where ``augmentations`` is ``None`` if no augmentation pipeline was
		saved in the checkpoint.

	Example:
		Resume without augmentations::

			model, noise_scheduler, optimizer, lr_scheduler, _ = load_checkpoint(
				"checkpoints/checkpoint-epoch-10"
			)
			train(my_dataset, model, noise_scheduler=noise_scheduler,
				optimizer=optimizer, lr_scheduler=lr_scheduler)

		Resume with augmentations::

			model, noise_scheduler, optimizer, lr_scheduler, augmentations = (
				load_checkpoint("checkpoints/checkpoint-epoch-10")
			)
			dataset.augmentations = augmentations
			train(my_dataset, model, noise_scheduler=noise_scheduler,
				optimizer=optimizer, lr_scheduler=lr_scheduler)
	"""
	model = AutoModel.from_pretrained(ckpt_path)

	with open(os.path.join(ckpt_path, "noise_scheduler.pkl"), "rb") as f:
		noise_scheduler = pickle.load(f)

	with open(os.path.join(ckpt_path, "optimizer.pkl"), "rb") as f:
		optimizer = pickle.load(f)

	with open(os.path.join(ckpt_path, "lr_scheduler.pkl"), "rb") as f:
		lr_scheduler = pickle.load(f)

	augmentations_path = os.path.join(ckpt_path, "augmentations.pkl")
	if os.path.exists(augmentations_path):
		with open(augmentations_path, "rb") as f:
			augmentations = pickle.load(f)
	else:
		augmentations = None

	return model, noise_scheduler, optimizer, lr_scheduler, augmentations


def train(
	dataset,
	model,
	*,
	noise_scheduler=None,           # None → DDPMScheduler(num_train_timesteps=1000)
	optimizer=None,                 # None → AdamW
	lr_scheduler=None,              # None → ConstantLR()
	output_dir: str = "checkpoints",
	num_epochs: int = 50,
	batch_size: int = 16,
	shuffle: bool = True,
	checkpoint_every_n_epochs: int = 5,
	mixed_precision: str = "fp16",
	gradient_accumulation_steps: int = 1,
	dataloader_num_workers: int = 4,
	max_grad_norm: float = 1.0,
	force_cpu: bool = False,
	verbose: bool = True,
):
	"""Train a diffusers diffusion model.

	To resume from a checkpoint, load it first with ``load_checkpoint()`` and
	pass the returned objects directly into this function. Augmentations are
	expected to be built into the dataset object directly (e.g. via
	``ArrayDataset.augmentations``), and are checkpointed automatically if the
	dataset exposes an ``augmentations`` attribute.

	The model's forward call is dispatched automatically: if the batch contains
	``"labels"``, they are passed as a keyword argument (for
	``DiTTransformer2DModel``); otherwise the model is called with positional
	``hidden_states`` and ``timestep`` arguments only (for ``UNet2DModel`` and
	similar).

	Args:
		dataset (torch.utils.data.Dataset): Dataset returning dicts with an
			``"images"`` key containing tensors in ``[-1, 1]``, and an optional
			``"labels"`` key (LongTensor of shape ``(batch_size,)``) for
			class-conditional DiT training. Augmentations should be applied
			inside the dataset's ``__getitem__``.
		model (nn.Module): Pre-instantiated diffusers model (e.g.
			``UNet2DModel``, ``DiTTransformer2DModel``).
		noise_scheduler (optional): Pre-instantiated diffusers noise scheduler.
			Defaults to ``DDPMScheduler(num_train_timesteps=1000)``.
		optimizer (torch.optim.Optimizer, optional): Optimizer for ``model``.
			Defaults to ``AdamW`` with PyTorch default lr.
		lr_scheduler (optional): A ``torch.optim.lr_scheduler`` instance.
			Defaults to ``ConstantLR(factor=1.0, total_iters=0)``, i.e. a
			fixed learning rate for the entire run.
		output_dir (str): Root directory for checkpoints and TensorBoard logs.
			Defaults to ``"checkpoints"``.
		num_epochs (int): Total number of training epochs. Defaults to ``50``.
		batch_size (int): Per-device batch size. Defaults to ``16``.
		shuffle (bool): Shuffle the dataset each epoch. Defaults to ``True``.
		checkpoint_every_n_epochs (int): Save a checkpoint every this many
			epochs. A final checkpoint is always saved at the last epoch.
			Defaults to ``5``.
		mixed_precision (str): AMP dtype passed to ``Accelerator`` — one of
			``"no"``, ``"fp16"``, or ``"bf16"``. Defaults to ``"fp16"``.
		gradient_accumulation_steps (int): Number of gradient accumulation steps
			before an optimizer update. Defaults to ``1``.
		dataloader_num_workers (int): Worker processes for the ``DataLoader``.
			Defaults to ``4``.
		max_grad_norm (float): Maximum gradient norm for clipping. Defaults to
			``1.0``.
		force_cpu (bool): Force the model to the CPU even if CUDA available
		verbose (bool): Print training progress and checkpoint messages.
			Defaults to ``True``.

	Example:
		Train a UNet from scratch with all defaults::

			from diffusers import UNet2DModel
			model = UNet2DModel(sample_size=64, in_channels=3, out_channels=3)
			train(my_dataset, model)

		Train a class-conditional DiT::

			from diffusers import DiTTransformer2DModel
			model = DiTTransformer2DModel(
				num_attention_heads=16,
				attention_head_dim=72,
				in_channels=4,
				sample_size=32,
				num_embeds_ada_norm=1000,
			)
			# dataset must return dicts with "images" and "labels" keys
			train(my_dataset, model)

		Resume from a checkpoint::

			model, noise_scheduler, optimizer, lr_scheduler, augmentations = (
				load_checkpoint("checkpoints/checkpoint-epoch-10")
			)
			dataset.augmentations = augmentations
			train(
				my_dataset,
				model,
				noise_scheduler=noise_scheduler,
				optimizer=optimizer,
				lr_scheduler=lr_scheduler,
			)
	"""
	# ------------------------------------------------------------------ #
	# 1.  Defaults                                                         #
	# ------------------------------------------------------------------ #
	if noise_scheduler is None:
		noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

	if optimizer is None:
		optimizer = torch.optim.AdamW(model.parameters())

	if lr_scheduler is None:
		lr_scheduler = ConstantLR(optimizer, factor=1.0, total_iters=0)

	# ------------------------------------------------------------------ #
	# 2.  Accelerator                                                      #
	# ------------------------------------------------------------------ #
	accelerator = Accelerator(
		mixed_precision=mixed_precision,
		gradient_accumulation_steps=gradient_accumulation_steps,
		log_with="tensorboard",
		project_dir=os.path.join(output_dir, "logs"),
		cpu=force_cpu,
	)

	# ------------------------------------------------------------------ #
	# 3.  DataLoader                                                       #
	# ------------------------------------------------------------------ #
	dataloader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=dataloader_num_workers,
		pin_memory=True,
		persistent_workers=dataloader_num_workers > 0,
	)

	# ------------------------------------------------------------------ #
	# 4.  Hand everything to Accelerate                                    #
	# ------------------------------------------------------------------ #
	model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
		model, optimizer, dataloader, lr_scheduler
	)
	# ------------------------------------------------------------------ #
	# 5.  Training loop                                                    #
	# ------------------------------------------------------------------ #
	os.makedirs(output_dir, exist_ok=True)
	global_step = 0

	metrics = {
		"loss": [],
		"times": [],
		"epoch_loss": [],
		"epoch_times": [],
	}

	for epoch in range(num_epochs):
		progress = tqdm(
			dataloader,
			desc=f"Epoch {epoch}/{num_epochs - 1}",
			disable=not verbose or not accelerator.is_local_main_process,
		)

		model.train()
		epoch_loss = 0.0
		epoch_time = 0.0

		for batch in progress:
			images = batch["images"]
			labels = batch.get("labels", None)

			t0 = time.time()

			with accelerator.accumulate(model):
				# --- forward diffusion ----------------------------------
				noise = torch.randn_like(images)
				timesteps = torch.randint(
					0,
					noise_scheduler.config.num_train_timesteps,
					(images.shape[0],),
					device=images.device,
				).long()
				noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

				# --- predict noise --------------------------------------
				with accelerator.autocast():
					if labels is not None:
						noise_pred = model(
							noisy_images,
							timestep=timesteps,
							class_labels=labels,
							return_dict=False,
						)[0]
					else:
						noise_pred = model(
							noisy_images,
							timesteps,
							return_dict=False,
						)[0]

					loss = F.mse_loss(noise_pred, noise)

				# --- backward -------------------------------------------
				accelerator.backward(loss)
				if accelerator.sync_gradients:
					accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

				optimizer.step()
				lr_scheduler.step()
				optimizer.zero_grad(set_to_none=True)

			batch_time = time.time() - t0
			batch_loss = loss.detach().item()

			global_step += 1
			epoch_loss += batch_loss
			epoch_time += batch_time

			metrics["loss"].append(batch_loss)
			metrics["times"].append(batch_time)

			progress.set_postfix(loss=batch_loss, lr=lr_scheduler.get_last_lr()[0])

		avg_loss = epoch_loss / len(dataloader)
		metrics["epoch_loss"].append(avg_loss)
		metrics["epoch_times"].append(epoch_time)

		accelerator.log({"train_loss": avg_loss, "epoch": epoch}, step=global_step)
		if verbose:
			accelerator.print(f"Epoch {epoch} — avg loss: {avg_loss:.4f}")

		# ---------------------------------------------------------------- #
		# 6.  Checkpointing                                                 #
		# ---------------------------------------------------------------- #
		if (epoch + 1) % checkpoint_every_n_epochs == 0 or epoch == num_epochs - 1:
			if accelerator.is_main_process:
				ckpt_save_path = os.path.join(output_dir, f"checkpoint-epoch-{epoch}")
				accelerator.save_state(ckpt_save_path)
				accelerator.unwrap_model(model).save_pretrained(ckpt_save_path)
				with open(os.path.join(ckpt_save_path, "optimizer.pkl"), "wb") as f:
					pickle.dump(optimizer.optimizer, f)
				with open(os.path.join(ckpt_save_path, "noise_scheduler.pkl"), "wb") as f:
					pickle.dump(noise_scheduler, f)
				with open(os.path.join(ckpt_save_path, "lr_scheduler.pkl"), "wb") as f:
					pickle.dump(lr_scheduler.scheduler, f)
				if hasattr(dataset, "augmentations") and dataset.augmentations is not None:
					with open(os.path.join(ckpt_save_path, "augmentations.pkl"), "wb") as f:
						pickle.dump(dataset.augmentations, f)
				if verbose:
					accelerator.print(f"Checkpoint saved → {ckpt_save_path}")

	accelerator.end_training()
	return metrics

