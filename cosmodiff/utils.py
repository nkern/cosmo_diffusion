import os
import glob
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
        # CUDA tensors cannot be shared across DataLoader worker processes (fork
        # cannot inherit the CUDA context). Move to CPU; the training loop
        # (via accelerate) handles GPU placement per-batch.
        self.arrays = arrays.cpu()
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
    log: bool = False,
    norm: str | None = None,
    two_dim: bool = True,
    zthin: int = 1,
    n_samples: int | None = None,
    seed: np.random.Generator | None = None,
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
        norm (str): Normalize images via "min-max" scaling ``[-1, 1]``,
            or "center-scale".
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

    if norm is not None:
        if norm == 'center-scale':
            images = center_scale_norm(images)
        elif norm == 'min-max':
            images = minmax_norm(images)

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


def center_scale_norm(x: torch.Tensor):
    """Center a tensor based on its median, and normalize by its absolute deviation.

    Args:
        x (torch.Tensor): Input tensor of any shape.

    Returns:
        torch.Tensor: scaled tensor
        float: avg
        float: std
    """
    # center
    avg = x.mean()
    x -= avg

    x /= x.abs().max()

    return x


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

    Example::

        with open("config.yaml") as f:
            config = yaml.safe_load(f)

        dataset = parse_config_data(config)
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
            imgs, lbls = load_data(img_path=p, label_path=lp, n_samples=ns, seed=sd, log=lg, **_load_kwargs)
            all_images.append(imgs)
            if lbls is not None:
                all_labels.append(lbls)

        images = torch.cat(all_images, dim=0)
        labels = torch.cat(all_labels, dim=0) if all_labels else None
    else:
        images, labels = load_data(
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

    return ArrayDataset(images, labels=labels, augmentations=augmentations)


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

    Produces four plots:
        - Batch loss over training steps
        - Epoch loss over epochs
        - Learning rate over epochs
        - Epoch wall time

    matplotlib is imported lazily inside this function so it is not a strict
    dependency of cosmodiff. If plotting on a remote server with no display,
    use ``save_dir`` and set ``show=False``.

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
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(metrics["loss"], alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Batch Loss")
    ax.set_yscale('log')
    fig.tight_layout()
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, "batch_loss.png"), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    # --- epoch loss -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(metrics["epoch_loss"], marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Epoch Loss")
    ax.set_yscale('log')
    fig.tight_layout()
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, "epoch_loss.png"), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    # --- learning rate --------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(metrics["epoch_lr"], marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    fig.tight_layout()
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, "learning_rate.png"), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    # --- epoch times ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(metrics["epoch_times"], marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Time (s)")
    ax.set_title("Epoch Wall Time")
    fig.tight_layout()
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, "epoch_times.png"), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


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
