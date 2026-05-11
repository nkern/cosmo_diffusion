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
from typing import Optional, Callable

from .transform import Normalization, MultiNormalization, Transform, MultiTransform


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
    img_path: str | np.ndarray | list[str],
    img_read_fn: Callable | list[Callable],
    device: str | None = None,
    dtype: torch.dtype | None = None,
    label_path: str | np.ndarray | list[str] | None = None,
    label_read_fn: Optional[Callable | list[Callable]] = None,
    reshape: str | None = '2d',
    zthin: int | list[int] = 1,
    n_samples: int | list[int] | None = None,
    seed: np.random.Generator | None = None,
    normalization: str | list[str] | None = None,
    norm_kwargs: dict | list[dict] | None = None,
    transform: list[str] | list[list[str]] | None = None,
    read_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Load images and optionally labels into tensors ready for ``ArrayDataset``.

    Both ``img_path`` and ``label_path`` can be either a filepath or a
    pre-loaded numpy array, allowing the function to be used in pipelines
    where data is already in memory.

    Input arrays are assumed to be of shape ``(Nbatch, Nz, Nx, Ny)``.

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
        reshape (str): If '2d' (default), treat each z-slice as independent 2D image
            and reshape to (batch, channel, H, W). If '3d', reshape to
            (batch, channel, D, H, W). If None, don't reshape.
        zthin (int): Thinning factor along the z axis, applied before
            reshaping when ``reshape='2d'``. A value of ``1`` applies no
            thinning. Defaults to ``1``.
        normalization (str): Normalize image pixel distribution.
            ['min-max', 'center-max']
        norm_kwargs: kwargs for pixel normalization function
        transform (list): list of data transforms e.g. ['log', 'fft2'].
            Ordered by operation. Note that 'log' always happens first if included.
        read_only (bool): if True, only read the data from disk
            to numpy array and return (no transform or normalization)

    Returns:
        dict: A dictionary with keys

            * ``'images'`` — ``torch.Tensor`` on ``device``.
            * ``'labels'`` — ``LongTensor`` of shape ``(N,)``, or ``None`` if
              ``label_path`` was not provided.
            * ``'norm'`` — fitted :class:`Normalization` instance, or ``None``
              if ``normalization`` was not requested.
            * ``'tform'`` — fitted :class:`Transform` instance, or ``None`` if
              ``transform`` was not requested.

    Example::

        import numpy as np

        def read_images(path):
            return np.load(path)

        out = load_data(
            img_path="images.npy",
            img_read_fn=read_images,
            device="cpu",
            dtype=torch.float32,
            reshape='2d',
            zthin=4,
        )
        images, labels = out['images'], out['labels']
    """
    # first check if feeding lists of filepaths
    if isinstance(img_path, (list, tuple)):

        # list of filepaths: assumes label_path is also list if provided
        n = len(img_path)
        if label_path is None:
            label_path = [label_path] * n
        assert isinstance(label_path, (list, tuple))
        img_read_fn = img_read_fn if isinstance(img_read_fn, (list, tuple)) else [img_read_fn] * n
        label_read_fn = label_read_fn if isinstance(label_read_fn, (list, tuple)) else [label_read_fn] * n
        zthin = zthin if isinstance(zthin, (list, tuple)) else [zthin] * n
        n_samples = n_samples if isinstance(n_samples, (list, tuple)) else [n_samples] * n
        seed = seed if isinstance(seed, (list, tuple)) else [seed] * n

        # two options: norm per filepath, or one norm for all data
        multi_norm = isinstance(normalization, (list, tuple))

        if not multi_norm:
            # one norm for all data.
            # first load and concat data, then apply transform / norm
            images, labels = [], []
            for img, ifn, lbl, lfn, zth, nsm, see in zip(
                img_path,
                img_read_fn,
                label_path,
                label_read_fn,
                zthin,
                n_samples,
                seed,
                ):
                # only load and reshape the data
                out = load_data( 
                    img,
                    img_read_fn=ifn,
                    label_path=lbl,
                    label_read_fn=lfn,
                    device=device,
                    dtype=dtype,
                    reshape=reshape,
                    zthin=zth,
                    n_samples=nsm,
                    seed=see,
                    read_only=True,
                )
                images.append(out['images'])
                if out['labels'] is not None:
                    labels.append(out['labels'])

            # concat
            images = torch.cat(images, dim=0)
            if len(labels) > 0:
                unq_labels = torch.cat([lbls[:1] for lbls in labels])
                labels = torch.cat(labels, dim=0)
            else:
                labels = None

            # pass through again to do transform / normalization
            output = load_data(
                images,
                img_read_fn=None,  # images already loaded
                label_path=labels,
                device=device,
                dtype=dtype,
                reshape=None,
                normalization=normalization,
                norm_kwargs=norm_kwargs,
                transform=transform,
            )

            return output

        else:
            # one norm for each dataset.
            # load each data, apply tform / norm, then concat.
            # note that this requires labels be supplied.

            # modify transform to be list of list if needed
            if transform is not None:
                # transform is either ['log', 'fft2'] or [['log'], None, ['log', 'fft2']]
                if isinstance(transform[0], str):
                    transform = [transform] * n

            images, labels, norms, tforms = [], [], [], []
            for img, ifn, lbl, lfn, zth, nsm, see, nrm, nkw, trn in zip(
                img_path,
                img_read_fn,
                label_path,
                label_read_fn,
                zthin,
                n_samples,
                seed,
                normalization,
                norm_kwargs,
                transform
                ):
                out = load_data( 
                    img,
                    img_read_fn=ifn,
                    label_path=lbl,
                    label_read_fn=lfn,
                    device=device,
                    dtype=dtype,
                    reshape=reshape,
                    zthin=zth,
                    n_samples=nsm,
                    seed=see,
                    normalization=nrm,
                    norm_kwargs=nkw,
                    transform=trn,
                    read_only=False,
                )
                images.append(out['images'])
                if out['labels'] is not None:
                    labels.append(out['labels'])
                if out['tform'] is not None:
                    tforms.append(out['tform'])
                if out['norm'] is not None:
                    norms.append(out['norm'])

            # concat
            images = torch.cat(images, dim=0)
            if len(labels) > 0:
                unq_labels = torch.cat([lbls[0] for lbls in labels])
                labels = torch.cat(labels, dim=0)
            else:
                labels = None

            if len(tforms) > 0:
                assert len(labels) > 0
                tform = MultiTransform(unq_labels, tforms)
            else:
                tform = None

            if len(norms) > 0:
                assert len(labels) > 0
                norm = MultiNormalization(unq_labels, norms)
            else:
                norm = None

            output = {
                'images': images,
                'labels': labels,
                'norm': norm,
                'tform': tform,
            }

            return output

    else:
        # assume img_path and label_path are single paths
        # --- images ---------------------------------------------------------
        if isinstance(img_path, (np.ndarray, torch.Tensor)):
            images = img_path
        else:
            assert img_read_fn is not None
            img_read_fn = globals()[img_read_fn] if isinstance(img_read_fn, str) else img_read_fn
            images = img_read_fn(img_path)

        if n_samples is not None:
            if seed is None:
                idx = slice(n_samples)
            else:
                rng = np.random.default_rng(seed)
                idx = rng.choice(len(images), size=n_samples, replace=False)
            images = images[idx]
        else:
            if isinstance(images, np.ndarray) and not images.flags.writeable:
                # may be memory-mapped ndarray, so make a copy
                images = np.asarray(images, copy=True)

        images = torch.as_tensor(images, device=device, dtype=dtype)

        # --- labels ---------------------------------------------------------
        labels = None
        if label_path is not None:
            if isinstance(label_path, (np.ndarray, torch.Tensor)):
                labels = label_path
            else:
                assert label_read_fn is not None
                label_read_fn = globals()[label_read_fn] if isinstance(label_read_fn, str) else label_read_fn
                labels = label_read_fn(label_path)

            if n_samples is not None:
                labels = labels[idx]
            else:
                labels = labels[:]

            labels = torch.as_tensor(labels, device=device, dtype=torch.long)

        # --- reshape --------------------------------------------------------
        if reshape == '2d':
            images = images[:, ::zthin]
            n_slices_per_vol = images.shape[1]
            images = images.reshape(-1, 1, *images.shape[-2:])
            # broadcast labels per-slice so they remain 1:1 with the reshaped images
            if labels is not None:
                labels = labels.repeat_interleave(n_slices_per_vol, dim=0)
        elif reshape == '3d':
            images = images.unsqueeze(1)

        if read_only:
            return {'images': images, 'labels': labels}

    # --- transforms -----------------------------------------------------
    tform = None
    if transform is not None:
        from cosmodiff.transform import Transform
        if 'log' in transform:
            log = True
            transform = [t for t in transform if t != 'log']
        else:
            log = False
        tform = Transform(transform, log=log)

        images = tform(images)

    norm = None
    if normalization is not None:
        norm = Normalization(normalization, inplace=False, **(norm_kwargs or {}))
        images = norm(images)


    output = {
        'images': images,
        'labels': labels,
        'norm': norm,
        'tform': tform,
    }

    return output


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


def parse_config_model(config: dict):
    """Instantiate model, optimizer, noise_scheduler, and lr_scheduler from a
    parsed yaml config dict. Any missing keys return ``None``, which will
    trigger the corresponding default in ``train()``.

    Args:
        config (dict): Parsed yaml config, e.g. from ``yaml.safe_load()``.

    Returns:
        dict: A dictionary with keys ``'model'``, ``'optimizer'``,
        ``'noise_scheduler'``, ``'lr_scheduler'``.  Each entry is the
        corresponding instantiated object, or ``None`` if the config did
        not specify it (or its dependencies could not be built).

    Example::

        with open("config.yaml") as f:
            config = yaml.safe_load(f)

        out = parse_config_model(config)
        model = out['model']
        optimizer = out['optimizer']
        noise_scheduler = out['noise_scheduler']
        lr_scheduler = out['lr_scheduler']
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

    output = {
        'model': model,
        'optimizer': optimizer,
        'noise_scheduler': noise_scheduler,
        'lr_scheduler': lr_scheduler
    }

    return output


def parse_config_data(config: dict):
    """Load data and build an ``ArrayDataset`` from a parsed yaml config dict.

    Args:
        config (dict): Parsed yaml config, e.g. from ``yaml.safe_load()``.

    Returns:
        dict: A dictionary with keys

            * ``'data'`` — :class:`ArrayDataset` ready to be passed to
              ``train()``.
            * ``'norm'`` — fitted :class:`Normalization` instance, or
              ``None`` if normalization was not requested.
            * ``'tform'`` — fitted :class:`Transform` instance, or ``None``
              if no ``transform`` list was given in the config.

    Example::

        with open("config.yaml") as f:
            config = yaml.safe_load(f)

        out = parse_config_data(config)
        dataset = out['data']
        norm = out['norm']
        tform = out['tform']
    """
    from cosmodiff import utils
    from cosmodiff.utils import ArrayDataset, load_data

    data_cfg = config["data"]
    global_cfg = config.get("global", {})

    dtype = getattr(torch, global_cfg.get("dtype", "float32"))
    device = "cpu" if data_cfg.get("keep_on_cpu", False) \
        else global_cfg.get("device", "cpu")


    out = load_data(
        img_path=data_cfg["img_path"],
        img_read_fn=data_cfg.get('img_read_fn', None),
        device=device,
        dtype=dtype,
        label_path=data_cfg.get("label_path", None),
        label_read_fn=data_cfg.get('label_read_fn', None),
        reshape=data_cfg.get("reshape", '2d'),
        zthin=data_cfg.get("zthin", 1),
        n_samples=data_cfg.get("n_samples", None),
        seed=data_cfg.get("seed", None),
        normalization=data_cfg.get('normalization', None),
        norm_kwargs=data_cfg.get('norm_kwargs', None),
        transform=data_cfg.get('transform', None),
    )

    augmentations = None
    if "augmentations" in config:
        from cosmodiff.augment import config_augmentations
        augmentations = config_augmentations(config["augmentations"])

    output = {
        'data': ArrayDataset(
            out['images'], labels=out['labels'], augmentations=augmentations,
        ),
        'norm': out['norm'],
        'tform': out['tform'],
    }

    return output


def npy_read_fn(fname):
    return np.load(fname, mmap_mode='r')


def txt_read_fn(fname):
    return np.loadtxt(fname)


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
        result = train(dataset, model)
        plot_metrics(result['metrics'], show=True)

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


def plot_ema_profiles(profiles: dict, ax=None):
    """Plot per-checkpoint and synthesized EMA weight profiles.

    Takes the dict returned by ``cosmodiff.optim.compute_ema_profiles`` and
    produces a figure analogous to Fig. 4 of Karras et al. 2024.

    Args:
        profiles: output of ``compute_ema_profiles``.
        ax: existing Axes to draw on; creates a new figure if ``None``.

    Returns:
        matplotlib Axes.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()

    epochs = profiles['epochs']

    for t_i, s_j, w in profiles['basis']:
        ax.plot(epochs, w, color='steelblue', alpha=0.4, linewidth=0.8,
                label=rf'checkpoint epoch {t_i}, $\sigma_{{rel}}$={s_j}')

    ax.plot(epochs, profiles['target'], 'k--', linewidth=1.5,
            label=r'target profile')
    ax.plot(epochs, profiles['synthesized'], 'r-', linewidth=2,
            label=r'synthesized profile')

    ax.set_xlabel('Training epoch')
    ax.set_ylabel('EMA weight (normalized)')
    ax.legend(fontsize=7)

    return ax


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
