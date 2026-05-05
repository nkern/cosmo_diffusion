import os
import glob
import pickle
import importlib
import inspect
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
    def __init__(self,
        arrays: torch.Tensor,
        labels: torch.Tensor = None,
        augmentations: callable = None,
    ):
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
    device: str | None = None,
    dtype: torch.dtype | None = None,
    label_path: str | np.ndarray | None = None,
    label_read_fn: Optional[callable] = None,
    two_dim: bool = True,
    zthin: int = 1,
    n_samples: int | None = None,
    seed: np.random.Generator | None = None,
    log: bool = False,
    normalization: str | None = None,
    norm_kwargs: {} | None = None,
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
        two_dim (bool): If ``True``, reshape the data to treat each z-slice
            as an independent 2D image. If ``False``, treat each sample as a
            3D volume. Defaults to ``True``.
        zthin (int): Thinning factor along the z axis, applied before
            reshaping when ``two_dim=True``. A value of ``1`` applies no
            thinning. Defaults to ``1``.
        log (bool): Apply a log transform to images before normalization.
            Defaults to ``False``.
        normalization (str): Normalize image pixel distribution.
            ['min-max', 'center-max']
        norm_kwargs: kwargs for pixel normalization function

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

    if n_samples is not None:
        if seed is None:
            idx = slice(n_samples)
        else:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(images), size=n_samples, replace=False)
        images = images[idx]
    else:
        # images may be a memory-mapped array, so slice to instantiate it
        images = images[:]

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

        if n_samples is not None:
            labels = labels[idx]
        else:
            labels = labels[:]

        labels = torch.as_tensor(labels, dtype=torch.long)

    # --- transforms -----------------------------------------------------
    if log:
        images = images.log()

    norm = None
    if normalization is not None:
        norm = Normalization(normalization, inplace=False, **norm_kwargs)
        images = norm(images)

    # --- reshape --------------------------------------------------------
    if two_dim:
        images = images[:, ::zthin]
        images = images.reshape(-1, 1, *images.shape[-2:])
    else:
        images = images.unsqueeze(1)

    return images, labels, norm


def _import_class(qualified_name: str):
    module_name, class_name = qualified_name.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def _get_lr_scheduler_kwargs(scheduler) -> dict:
    sig = inspect.signature(scheduler.__class__.__init__)
    kwargs = {}
    for name in sig.parameters:
        if name in ("self", "optimizer", "last_epoch", "verbose"):
            continue
        if hasattr(scheduler, name):
            kwargs[name] = getattr(scheduler, name)
    return kwargs


def load_checkpoint(ckpt_path: str):
    """Reconstruct model, noise_scheduler, optimizer, lr_scheduler, and
    optionally an augmentation pipeline from a saved checkpoint directory
    produced by ``train()``.

    The returned objects are freshly constructed from the checkpoint config
    and are ready to be passed directly into ``train()``.  Pass
    ``resume_from_checkpoint=ckpt_path`` to ``train()`` so it can call
    ``accelerator.load_state()`` after ``accelerator.prepare()`` to restore
    optimizer moments, the grad scaler, and RNG state.

    Args:
        ckpt_path (str): Path to a checkpoint directory produced by ``train()``.

    Returns:
        tuple: ``(model, noise_scheduler, optimizer, lr_scheduler, augmentations)``
        where ``augmentations`` is ``None`` if no augmentation pipeline was
        saved in the checkpoint.

    Example:
        Resume training::

            model, noise_scheduler, optimizer, lr_scheduler, augmentations = (
                load_checkpoint("checkpoints/checkpoint-epoch-10")
            )
            if augmentations is not None:
                dataset.augmentations = augmentations
            train(
                my_dataset, model,
                noise_scheduler=noise_scheduler,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                resume_from_checkpoint="checkpoints/checkpoint-epoch-10",
            )
    """
    model = AutoModel.from_pretrained(ckpt_path)

    with open(os.path.join(ckpt_path, "checkpoint_config.yaml")) as f:
        cfg = yaml.safe_load(f)

    noise_scheduler_cls = _import_class(cfg["noise_scheduler"]["class"])
    noise_scheduler = noise_scheduler_cls.from_pretrained(ckpt_path)

    optimizer_cls = _import_class(cfg["optimizer"]["class"])
    optimizer = optimizer_cls(model.parameters())

    lr_scheduler_cls = _import_class(cfg["lr_scheduler"]["class"])
    lr_scheduler = lr_scheduler_cls(optimizer, **cfg["lr_scheduler"]["kwargs"])

    augmentations_path = os.path.join(ckpt_path, "augmentations.pkl")
    if os.path.exists(augmentations_path):
        with open(augmentations_path, "rb") as f:
            augmentations = pickle.load(f)
    else:
        augmentations = None

    return model, noise_scheduler, optimizer, lr_scheduler, augmentations


def minmax_norm(
    x: torch.Tensor,
    xmin: float | None = None,
    xmax: float | None = None,
    inverse: bool = False,
    inplace: bool = False,
    **kwargs
) -> torch.Tensor:
    """Normalize a tensor to ``[-1, 1]`` via min-max scaling.

    Args:
        x (torch.Tensor): Input tensor of any shape.
        xmin (float): normalize by min, default is x.min()
        xmax (float): normalize by max, default is x.max()
        inverse (bool): if True, apply the inverse mapping (requires xmin, xmax)
        inplace (bool): edit tensor inplace, default is False

    Returns:
        torch.Tensor: Normalized tensor with values in ``[-1, 1]``.
        dict: normalization parameters
    """
    if not inplace:
        x = x.clone()
    if not inverse:
        if xmin is None:
            xmin = x.min()
        if xmax is None:
            xmax = x.max()
        x -= xmin
        x *= 2 / xmax
        x -= 1

    else:
        assert xmin is not None
        assert xmax is not None
        x = (x + 1) / 2 * xmax + xmin

    return x, {'xmin': xmin, 'xmax': xmax}


def center_max_norm(
    x: torch.Tensor,
    center: float | None = None,
    xmax: float | None = None,
    inverse: bool | None = False,
    inplace: bool = False,
    **kwargs
):
    """Center a tensor based on its average, and normalize by its absolute deviation.

    Args:
        x (torch.Tensor): Input tensor of any shape.
        center (float): average centering to subtract off
        xmax (float): abs-max normalization to divide by
        inverse (bool): apply inverse operation, requires center and xmax
        inplace (bool): If True, edit inplace. 

    Returns:
        torch.Tensor: scaled tensor
        dict: normalization parameters
    """
    if not inplace:
        x = x.clone()

    if not inverse:
        # center
        if center is None:
            center = x.mean()
        x -= center

        # scale by max-abs
        if xmax is None:
            xmax = x.abs().max()
        x /= xmax

    else:
        assert xmax is not None
        assert center is not None
        x *= xmax

    return x, {'center': center, 'xmax': xmax}


def tanh_norm(x, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0, mu=0.0, inverse=False, **kwargs):
    """
    A unified function for a dual-saturating tanh transform.
    
    Args:
        x (tensor): Input tensor
        alpha (float): Positive saturation limit (upper bound).
        beta (float): Negative saturation limit (magnitude of lower bound).
        gamma (float): Multiplicative gain for the positive side.
        delta (float): Multiplicative gain for the negative side.
        mu (float): Mean shift / center point.
        inverse (bool): Toggle between forward and inverse operations.

    Returns:
        torch.Tensor: normalized data
        dict: normalization parameters
    """
    if not inverse:
        # Forward: x -> y
        x_shifted = data - mu
        pos = alpha * torch.tanh((gamma * x_shifted) / alpha)
        neg = beta * torch.tanh((delta * x_shifted) / beta)
        y = torch.where(x_shifted >= 0, pos, neg)
    else:
        # Inverse: y -> x
        # Note: data (y) should be clamped within (-beta, alpha) for stability
        y = torch.clamp(data, -beta + 1e-9, alpha - 1e-9)
        pos_inv = (alpha * torch.atanh(y / alpha)) / gamma
        neg_inv = (beta * torch.atanh(y / beta)) / delta
        y =  torch.where(y >= 0, pos_inv, neg_inv) + mu

    params = {'mu': mu, 'alpha': alpha, 'beta': beta, 'delta': delta, 'gamma': gamma}
    return y, params


class Normalization(torch.nn.Module):
    """A data normalization object"""

    def __init__(self, method: str, inplace: bool = False, **kwargs):
        self.method = method
        self.inplace = inplace
        self.kwargs = kwargs

    def forward(self, x, inverse=False):
        if self.method in ['minmax', 'min-max']:
            x, kw = minmax_norm(x, inplace=self.inplace, inverse=inverse, **self.kwargs)
            self.kwargs.update(kw)

        elif self.method in ['centermax', 'center-max']:
            x, kw = center_max_norm(x, inplace=self.inplace, inverse=inverse, **self.kwargs)
            self.kwargs.update(kw)

        elif self.method in ['tanh']:
            x, kw = tanh_norm(x, inplace=self.inplace, inverse=inverse, **self.kwargs)
            self.kwargs.update(kw)

    def inverse(self, x):
        return self.forward(x, inverse=True)


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

    global_cfg = config.get("global", {})
    device = global_cfg.get("device", "cpu")

    # --- model ----------------------------------------------------------
    model = None
    if "model" in config:
        model_cls = getattr(diffusers, config["model"]["class"])
        model = model_cls(**config["model"].get("kwargs", {}))
        model.to(device)

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
        Normalization: object that (un) normalizes the data

    Example::

        with open("config.yaml") as f:
            config = yaml.safe_load(f)

        dataset, norm = parse_config_data(config)
    """
    import cosmodiff.utils as utils_module
    from cosmodiff.utils import ArrayDataset, load_data

    data_cfg = config["data"]
    global_cfg = config.get("global", {})

    dtype = getattr(torch, global_cfg.get("dtype", "float32"))
    device = "cpu" if data_cfg.get("keep_on_cpu", False) \
        else global_cfg.get("device", "cpu")

    img_read_fn = getattr(utils_module, data_cfg["img_read_fn"])

    label_path = data_cfg.get("label_path", None)
    label_read_fn = None
    if "label_read_fn" in data_cfg:
        label_read_fn = getattr(utils_module, data_cfg["label_read_fn"])

    img_path = data_cfg["img_path"]
    n_samples = data_cfg.get("n_samples", None)
    seed = data_cfg.get("seed", None)
    log = data_cfg.get("log", False)

    _load_kwargs = dict(
        img_read_fn=img_read_fn,
        device=device,
        dtype=dtype,
        label_read_fn=label_read_fn,
        norm=data_cfg.get("norm", 'center-scale'),
        two_dim=data_cfg.get("two_dim", True),
        zthin=data_cfg.get("zthin", 1),
    )

    if isinstance(img_path, (list, tuple)):
        n = len(img_path)
        label_paths = label_path if isinstance(label_path, (list, tuple)) else [label_path] * n
        n_samples_list = n_samples if isinstance(n_samples, (list, tuple)) else [n_samples] * n
        seeds_list = seed if isinstance(seed, (list, tuple)) else [seed] * n
        log_list = log if isinstance(log, (list, tuple)) else [log] * n

        all_images, all_labels = [], []
        for p, lp, ns, sd, lg in zip(img_path, label_paths, n_samples_list, seeds_list, log_list):
            imgs, lbls, norm = load_data(img_path=p, label_path=lp, n_samples=ns, seed=sd, log=lg, **_load_kwargs)
            all_images.append(imgs)
            if lbls is not None:
                all_labels.append(lbls)

        images = torch.cat(all_images, dim=0)
        labels = torch.cat(all_labels, dim=0) if all_labels else None
    else:
        images, labels, norm = load_data(
            img_path=img_path,
            label_path=label_path,
            n_samples=n_samples,
            seed=seed,
            log=log,
            **_load_kwargs,
        )

    augmentations = None
    if "augmentations" in config:
        from cosmodiff.augment import config_augmentations
        augmentations = config_augmentations(config["augmentations"])

    return ArrayDataset(images, labels=labels, augmentations=augmentations), norm


def npy_read_fn(fname):
    return np.load(fname, mmap_mode='r')


def read_logs(output_dir: str) -> dict:
    """Extract training metrics from TensorBoard logs produced by ``optim.train()``.

    Args:
        output_dir (str): Root output directory passed to ``train()``, which
            contains a ``logs/`` subdirectory with TensorBoard event files.

    Returns:
        dict: Dictionary with keys ``"epoch_loss"`` and ``"epoch"`` mapping to
        lists of scalar values in the order they were logged.

    Example::

        metrics = read_logs("checkpoints")
        print(metrics["epoch_loss"])
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    log_dir = os.path.join(output_dir, "logs")
    ea = EventAccumulator(log_dir)
    ea.Reload()

    metrics = {}
    for tag in ea.Tags()["scalars"]:
        events = ea.Scalars(tag)
        metrics[tag] = [e.value for e in events]

    return metrics


def write_metrics(metrics: dict, filepath: str) -> None:
    """Write a training metrics dictionary to a JSON file.

    Args:
        metrics (dict): Metrics dictionary returned by ``train()``.
        filepath (str): Path to the output JSON file.

    Example::

        write_metrics(metrics, "output/metrics_50.json")
    """
    import json
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)


def read_metrics(filepath: str) -> dict:
    """Read a training metrics dictionary from a JSON file.

    Args:
        filepath (str): Path to a JSON file produced by ``write_metrics()``.

    Returns:
        dict: Metrics dictionary with keys ``"loss"``, ``"times"``,
        ``"epoch_loss"``, ``"epoch_times"``, ``"epoch_lr"``.

    Example::

        metrics = read_metrics("output/metrics_50.json")
    """
    import json
    with open(filepath) as f:
        return json.load(f)


def plot_metrics(metrics: dict | str, save_dir: str = None, show: bool = False) -> None:
    """Plot training metrics from a dictionary or JSON file produced by
    ``write_metrics()``.

    Produces three plots:
        - Batch loss over training steps
        - Epoch loss over epochs
        - Learning rate over epochs

    Args:
        metrics (dict or str): Metrics dictionary returned by ``train()``, or
            a path to a JSON file produced by ``write_metrics()``.
        save_dir (str, optional): Directory to save plots. If ``None``, plots
            are not saved to disk. Defaults to ``None``.
        show (bool): Whether to call ``plt.show()`` after each plot. Useful in
            interactive sessions. Defaults to ``False``.

    Example::

        # from dict
        metrics = train(dataset, model)
        plot_metrics(metrics, show=True)

        # from file
        plot_metrics("output/metrics_epoch_49.json", save_dir="output/plots")
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if isinstance(metrics, str):
        metrics = read_metrics(metrics)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # --- batch loss -----------------------------------------------------
    fig1, ax = plt.subplots(figsize=(8, 4))
    ax.plot(metrics["loss"], alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Batch Loss")
    ax.set_yscale('log')
    fig1.tight_layout()
    if save_dir is not None:
        fig1.savefig(os.path.join(save_dir, "batch_loss.png"), dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    # --- epoch loss -----------------------------------------------------
    fig2, ax = plt.subplots(figsize=(8, 4))
    ax.plot(metrics["epoch_loss"], marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Epoch Loss")
    ax.set_yscale('log')
    fig2.tight_layout()
    if save_dir is not None:
        fig2.savefig(os.path.join(save_dir, "epoch_loss.png"), dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    # --- learning rate --------------------------------------------------
    fig3, ax = plt.subplots(figsize=(8, 4))
    ax.plot(metrics["epoch_lr"], marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale('log')
    fig3.tight_layout()
    if save_dir is not None:
        fig3.savefig(os.path.join(save_dir, "learning_rate.png"), dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    # --- epoch times ----------------------------------------------------
    #fig, ax = plt.subplots(figsize=(8, 4))
    #ax.plot(metrics["epoch_times"], marker="o")
    #ax.set_xlabel("Epoch")
    #ax.set_ylabel("Time (s)")
    #ax.set_title("Epoch Wall Time")
    #fig.tight_layout()
    #if save_dir is not None:
    #    fig.savefig(os.path.join(save_dir, "epoch_times.png"), dpi=150, bbox_inches="tight")
    #if show:
    #    plt.show()
    #plt.close(fig)

    return fig1, fig2, fig3


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Return the path to the latest checkpoint in ``output_dir``, or ``None``
    if no checkpoints exist.

    Args:
        output_dir (str): Directory to search for checkpoints.

    Returns:
        str or None: Path to the latest checkpoint directory.
    """
    pattern = os.path.join(output_dir, "checkpoint-epoch-*")
    checkpoints = sorted(
        glob.glob(pattern),
        key=lambda p: int(p.split("-")[-1]),
    )
    return checkpoints[-1] if checkpoints else None


def read_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config
