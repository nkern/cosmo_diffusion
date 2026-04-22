import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ConstantLR
from accelerate import Accelerator
import diffusers
from diffusers import AutoModel, DDPMScheduler
from tqdm.auto import tqdm
import time
import yaml
from typing import Optional


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


def load_data(
	img_path: str | np.ndarray,
	img_read_fn: callable,
	device: str,
	dtype: torch.dtype,
	label_path: str | np.ndarray | None = None,
	label_read_fn: Optional[callable] = None,
	log: bool = False,
	minmax: bool = True,
	two_dim: bool = True,
	zthin: int = 1,
) -> tuple[torch.Tensor, torch.Tensor | None]:
	"""Load images and optionally labels into tensors ready for ``ArrayDataset``.

	Both ``img_path`` and ``label_path`` can be either a filepath or a
	pre-loaded numpy array, allowing the function to be used in pipelines
	where data is already in memory.

	Input arrays are assumed to be of shape ``(Nbatch, Nz, Nx, Ny)``.

	If ``two_dim=True``, the z axis is optionally thinned by ``zthin`` and
	the array is reshaped to ``(Nbatch * Nz, 1, Nx, Ny)``, treating each
	z-slice as an independent 2D image.

	If ``two_dim=False``, ``zthin`` is ignored and the array is unsqueezed
	to ``(Nbatch, 1, Nz, Nx, Ny)`` treating each sample as a 3D volume
	with a single channel.

	Args:
		img_path (str or np.ndarray): Path to the image data file on disk, or
			a pre-loaded numpy array.
		img_read_fn (callable): User-provided function that accepts
			``img_path`` and returns a numpy array of shape
			``(Nbatch, Nz, Nx, Ny)``. Ignored if ``img_path`` is already a
			numpy array.
		device (str): Device to place the image tensor on, e.g. ``"cpu"``
			or ``"cuda"``.
		dtype (torch.dtype): Floating point dtype for the image tensor, e.g.
			``torch.float32``.
		label_path (str or np.ndarray, optional): Path to the label data file
			on disk, or a pre-loaded numpy array. Defaults to ``None``.
		label_read_fn (callable, optional): User-provided function that accepts
			``label_path`` and returns a numpy array of shape ``(N,)``. Ignored
			if ``label_path`` is already a numpy array. Defaults to ``None``.
		log (bool): Apply a log transform to images before normalization.
			Defaults to ``False``.
		minmax (bool): Normalize images to ``[-1, 1]`` via min-max scaling.
			Defaults to ``True``.
		two_dim (bool): If ``True``, reshape the data to treat each z-slice
			as an independent 2D image. If ``False``, treat each sample as a
			3D volume. Defaults to ``True``.
		zthin (int): Thinning factor along the z axis, applied before
			reshaping when ``two_dim=True``. A value of ``1`` applies no
			thinning. Defaults to ``1``.

	Returns:
		tuple: ``(images, labels)`` where ``images`` is a ``torch.Tensor`` on
		``device``, and ``labels`` is a LongTensor of shape ``(N,)`` or
		``None`` if ``label_path`` was not provided.

	Example::

		import numpy as np

		def read_images(path):
		    return np.load(path)

		images, labels = load_data(
		    img_path="images.npy",
		    img_read_fn=read_images,
		    device="cpu",
		    dtype=torch.float32,
		    two_dim=True,
		    zthin=4,
		)
	"""
	# --- images ---------------------------------------------------------
	if isinstance(img_path, np.ndarray):
		images = img_path
	else:
		images = img_read_fn(img_path)

	images = torch.as_tensor(images, device=device, dtype=dtype)

	# --- labels ---------------------------------------------------------
	labels = None
	if label_path is not None:
		if isinstance(label_path, np.ndarray):
			labels = label_path
		else:
			if label_read_fn is None:
				raise ValueError(
					"label_read_fn must be provided when label_path is a filepath."
				)
			labels = label_read_fn(label_path)

		labels = torch.as_tensor(labels, dtype=torch.long)

	# --- transforms -----------------------------------------------------
	if log:
		images = images.log()

	if minmax:
		images = images - images.min()
		images = images / images.max() * 2 - 1.0

	# --- reshape --------------------------------------------------------
	if two_dim:
		images = images[:, ::zthin]
		images = images.reshape(-1, 1, *images.shape[-2:])
	else:
		images = images.unsqueeze(1)

	return images, labels


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


def minmax_norm(x: torch.Tensor) -> torch.Tensor:
	"""Normalize a tensor to ``[-1, 1]`` via min-max scaling.

	Args:
		x (torch.Tensor): Input tensor of any shape.

	Returns:
		torch.Tensor: Normalized tensor with values in ``[-1, 1]``.
	"""
	x = x - x.min()
	x = x / x.max()
	return x * 2 - 1


def center_scale_norm(x: torch.Tensor, scale: int = 10):
	"""Center a tensor based on its median, and normalize by its scale

	Args:
		x (torch.Tensor): Input tensor of any shape.
		scale (int): Number of standard deviations to scale by

	Returns:
		torch.Tensor: scaled tensor
		float: avg
		float: std
	"""
	# center
	avg = x.median()
	x -= avg

	# norm
	std = x.std()
	x /= std * scale

	return x, avg, std


def parse_config_model(config: dict):
	"""Instantiate model, optimizer, noise_scheduler, and lr_scheduler from a
	parsed yaml config dict. Any missing keys return ``None``, which will
	trigger the corresponding default in ``train()``.

	Args:
		config (dict): Parsed yaml config, e.g. from ``yaml.safe_load()``.

	Returns:
		tuple: ``(model, optimizer, noise_scheduler, lr_scheduler)`` where any
		missing component is ``None``.

	Example::

		with open("config.yaml") as f:
		    config = yaml.safe_load(f)

		model, optimizer, noise_scheduler, lr_scheduler = parse_model(config)
	"""
	import torch.optim.lr_scheduler as lr_schedulers

	# --- model ----------------------------------------------------------
	model = None
	if "model" in config:
		model_cls = getattr(diffusers, config["model"]["class"])
		model = model_cls(**config["model"].get("kwargs", {}))

	# --- optimizer ------------------------------------------------------
	optimizer = None
	if "optimizer" in config and model is not None:
		opt_cls = getattr(torch.optim, config["optimizer"]["class"])
		optimizer = opt_cls(model.parameters(), **config["optimizer"].get("kwargs", {}))

	# --- noise scheduler ------------------------------------------------
	noise_scheduler = None
	if "noise_scheduler" in config:
		scheduler_cls = getattr(diffusers, config["noise_scheduler"]["class"])
		noise_scheduler = scheduler_cls(**config["noise_scheduler"].get("kwargs", {}))

	# --- lr scheduler ---------------------------------------------------
	lr_scheduler = None
	if "lr_scheduler" in config and optimizer is not None:
		lr_cls = getattr(lr_schedulers, config["lr_scheduler"]["class"])
		lr_scheduler = lr_cls(optimizer, **config["lr_scheduler"].get("kwargs", {}))

	return model, optimizer, noise_scheduler, lr_scheduler


def parse_config_data(config: dict):
	"""Load data and build an ``ArrayDataset`` from a parsed yaml config dict.

	Read functions are resolved by name from ``cosmodiff.utils``. Any callable
	in that module can be referenced by name in the yaml config.

	Args:
		config (dict): Parsed yaml config, e.g. from ``yaml.safe_load()``.

	Returns:
		ArrayDataset: Dataset ready to be passed to ``train()``.

	Example::

		with open("config.yaml") as f:
		    config = yaml.safe_load(f)

		dataset = parse_config_data(config)
	"""
	import cosmodiff.utils as utils_module
	from cosmodiff.utils import ArrayDataset, load_data

	data_cfg = config["data"]

	dtype = getattr(torch, data_cfg.get("dtype", "float32"))
	device = data_cfg.get("device", "cpu")

	img_read_fn = getattr(utils_module, data_cfg["img_read_fn"])

	label_path = data_cfg.get("label_path", None)
	label_read_fn = None
	if "label_read_fn" in data_cfg:
		label_read_fn = getattr(utils_module, data_cfg["label_read_fn"])

	images, labels = load_data(
		img_path=data_cfg["img_path"],
		img_read_fn=img_read_fn,
		device=device,
		dtype=dtype,
		label_path=label_path,
		label_read_fn=label_read_fn,
		log=data_cfg.get("log", False),
		minmax=data_cfg.get("minmax", True),
		two_dim=data_cfg.get("two_dim", True),
		zthin=data_cfg.get("zthin", 1),
	)

	augmentations = None
	if "augmentations" in config:
		from cosmodiff.augment import config_augmentations
		augmentations = config_augmentations(config["augmentations"])

	return ArrayDataset(images, labels=labels, augmentations=augmentations)


def npy_read_fn(fname):
	return np.load(fname)
