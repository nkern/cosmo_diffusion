import os
import pickle
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ConstantLR
from accelerate import Accelerator
from diffusers import AutoModel, DDPMScheduler
from tqdm.auto import tqdm
from typing import Callable, Optional
import time

from . import utils


def _to_yaml_safe(obj):
    """Recursively convert tuples to lists so yaml.safe_load can round-trip."""
    if isinstance(obj, dict):
        return {k: _to_yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_yaml_safe(v) for v in obj]
    return obj


def train(
    dataset,
    model=None,
    *,
    noise_scheduler=None,           # None → DDPMScheduler(num_train_timesteps=1000)
    optimizer=None,                 # None → AdamW
    lr_scheduler=None,              # None → ConstantLR()
    output_dir: str = "checkpoints",
    resume_from_checkpoint: Optional[str] = None,
    num_epochs: int = 50,
    batch_size: int = 16,
    shuffle: bool = True,
    checkpoint_every_n_epochs: int = 5,
    mixed_precision: str = "fp16",
    gradient_accumulation_steps: int = 1,
    dataloader_num_workers: int = 0,
    max_grad_norm: float = 1.0,
    force_cpu: bool = False,
    pin_memory: bool = False,
    verbose: bool = True,
):
    """Train a diffusers diffusion model.

    ``model`` is optional when ``resume_from_checkpoint`` is set — the model
    (and any unspecified scheduler/optimizer/augmentations) are loaded from the
    checkpoint automatically.  Augmentations are checkpointed automatically if
    the dataset exposes an ``augmentations`` attribute.

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
        model (nn.Module, optional): Pre-instantiated diffusers model (e.g.
            ``UNet2DModel``, ``DiTTransformer2DModel``).  May be omitted when
            ``resume_from_checkpoint`` is provided.
        noise_scheduler (optional): Pre-instantiated diffusers noise scheduler.
            Defaults to ``DDPMScheduler(num_train_timesteps=1000)``.
        optimizer (torch.optim.Optimizer, optional): Optimizer for ``model``.
            Defaults to ``AdamW`` with PyTorch default lr.
        lr_scheduler (optional): A ``torch.optim.lr_scheduler`` instance.
            Defaults to ``ConstantLR(factor=1.0, total_iters=0)``, i.e. a
            fixed learning rate for the entire run.
        output_dir (str): Root directory for checkpoints and TensorBoard logs.
            Defaults to ``"checkpoints"``.
        resume_from_checkpoint (str, optional): Path to a checkpoint directory
            produced by a previous call to ``train()``.  Objects not explicitly
            passed (model, noise_scheduler, optimizer, lr_scheduler) are loaded
            from the checkpoint.  After ``accelerator.prepare()`` the full
            training state (optimizer moments, grad scaler, RNG) is restored
            via ``accelerator.load_state()``.
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
            Defaults to ``0``.
        max_grad_norm (float): Maximum gradient norm for clipping. Defaults to
            ``1.0``.
        force_cpu (bool): Force the model to the CPU even if CUDA available
        pin_memory (bool): pin dataset memory if on CPU
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

        Resume from a checkpoint (model loaded automatically)::

            train(my_dataset, resume_from_checkpoint="checkpoints/checkpoint-epoch-0010")
    """
    # ------------------------------------------------------------------ #
    # 1.  Defaults / checkpoint loading                                    #
    # ------------------------------------------------------------------ #
    start_epoch = 0

    if model is None and resume_from_checkpoint is None:
        raise ValueError(
            "Either `model` or `resume_from_checkpoint` must be provided."
        )

    if resume_from_checkpoint is not None:
        model, noise_scheduler, optimizer, lr_scheduler, _aug = (
            utils.load_checkpoint(resume_from_checkpoint)
        )
        if isinstance(dataset, utils.ArrayDataset):
            dataset.augmentations = _aug

        start_epoch = int(resume_from_checkpoint.split('-')[-1]) + 1

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
    accelerator.init_trackers(project_name="cosmodiff")

    # Register hooks so save_state() delegates model serialisation to
    # save_pretrained() and load_state() restores via from_pretrained(),
    # avoiding a redundant second copy of the weights on disk.
    def _save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for m in models:
                m.save_pretrained(output_dir)
            weights.clear()

    def _load_model_hook(models, input_dir):
        for _ in range(len(models)):
            m = models.pop()
            loaded = m.__class__.from_pretrained(input_dir)
            m.register_to_config(**loaded.config)
            m.load_state_dict(loaded.state_dict())
            del loaded

    accelerator.register_save_state_pre_hook(_save_model_hook)
    accelerator.register_load_state_pre_hook(_load_model_hook)

    # ------------------------------------------------------------------ #
    # 3.  DataLoader                                                       #
    # ------------------------------------------------------------------ #
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=dataloader_num_workers,
        pin_memory=pin_memory,
        persistent_workers=dataloader_num_workers > 0,
    )

    # ------------------------------------------------------------------ #
    # 4.  Hand everything to Accelerate                                    #
    # ------------------------------------------------------------------ #
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    if resume_from_checkpoint is not None:
        accelerator.load_state(resume_from_checkpoint)

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
        "epoch_lr": [],
    }

    for epoch in range(start_epoch, start_epoch + num_epochs):
        progress = tqdm(
            dataloader,
            desc=f"Epoch {epoch}/{start_epoch + num_epochs - 1}",
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
                        pred = model(
                            noisy_images,
                            timestep=timesteps,
                            class_labels=labels,
                            return_dict=False,
                        )[0]
                    else:
                        pred = model(
                            noisy_images,
                            timesteps,
                            return_dict=False,
                        )[0]

                    prediction_type = noise_scheduler.config.prediction_type
                    if prediction_type == 'epsilon':
                        target = noise
                    elif prediction_type == 'v_prediction':
                        target = noise_scheduler.get_velocity(images, noise, timesteps)
                    elif prediction_type == 'sample':
                        target = images
                    else:
                        raise ValueError(f"Unsupported prediction_type: {prediction_type}")
                    loss = F.mse_loss(pred, target)

                # --- backward -------------------------------------------
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            lr_scheduler.step()
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
        metrics['epoch_lr'].append(lr_scheduler.get_last_lr()[0])

        accelerator.log({
            "train_loss": avg_loss,
            "epoch": epoch,
            "epoch_time": epoch_time,
            "learning_rate": lr_scheduler.get_last_lr()[0],
        }, step=global_step)
        if verbose:
            accelerator.print(f"Epoch {epoch} — avg loss: {avg_loss:.4f}")

        # ---------------------------------------------------------------- #
        # 6.  Checkpointing                                                 #
        # ---------------------------------------------------------------- #
        if (epoch + 1) % checkpoint_every_n_epochs == 0 or epoch == (start_epoch + num_epochs - 1):
            if accelerator.is_main_process:
                ckpt_save_path = os.path.join(output_dir, f"checkpoint-epoch-{epoch:04d}")

                # Noise scheduler config (needed by SchedulerClass.from_pretrained)
                noise_scheduler.save_pretrained(ckpt_save_path)

                # Class names and constructor kwargs for fresh reconstruction:
                # this is only needed when resuming training from a checkpoint.
                raw_opt = optimizer.optimizer
                raw_sched = lr_scheduler.scheduler
                ckpt_cfg = {
                    "noise_scheduler": {
                        "class": f"{noise_scheduler.__class__.__module__}.{noise_scheduler.__class__.__name__}",
                    },
                    "optimizer": {
                        "class": f"{raw_opt.__class__.__module__}.{raw_opt.__class__.__name__}",
                    },
                    "lr_scheduler": {
                        "class": f"{raw_sched.__class__.__module__}.{raw_sched.__class__.__name__}",
                        "kwargs": utils._get_lr_scheduler_kwargs(raw_sched),
                    },
                }
                with open(os.path.join(ckpt_save_path, "checkpoint_config.yaml"), "w") as f:
                    yaml.dump(_to_yaml_safe(ckpt_cfg), f)

                # Model weights (via hook) + optimizer moments + grad scaler + RNG
                accelerator.save_state(ckpt_save_path)

                if hasattr(dataset, "augmentations") and dataset.augmentations is not None:
                    with open(os.path.join(ckpt_save_path, "augmentations.pkl"), "wb") as f:
                        pickle.dump(dataset.augmentations, f)

                metrics_path = os.path.join(ckpt_save_path, "metrics.json")
                utils.write_metrics(metrics, metrics_path)

                if verbose:
                    accelerator.print(f"Checkpoint saved → {ckpt_save_path}")

    accelerator.end_training()
    return metrics


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    noise_scheduler,
    batch_size: int = 1,
    image_shape: tuple[int, ...] = (1, 64, 64),
    labels: Optional[torch.Tensor] = None,
    ddim_thinning: Optional[int] = None,
    renorm: Optional[Callable] = None,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Generate a batch of novel images via reverse diffusion.

    Compatible with DDPMScheduler and DDIMScheduler. The number of inference
    steps defaults to ``noise_scheduler.config.num_train_timesteps``, or can
    be reduced via ``ddim_thinning`` for faster DDIM sampling.

    Args:
        model: Trained diffusers model (UNet2DModel or DiTTransformer2DModel).
        noise_scheduler: Trained scheduler (DDPMScheduler, DDIMScheduler, or
            any compatible scheduler with ``set_timesteps`` and ``step``).
        batch_size (int): Number of images to generate.
        image_shape (tuple): Shape of each image ``(C, H, W)``.
        labels (array-like, optional): Class indices of length ``batch_size``.
            Required for class-conditional models (e.g. DiTTransformer2DModel);
            ignored otherwise.
        ddim_thinning (int, optional): Thinning factor relative to
            ``num_train_timesteps``. E.g. ``ddim_thinning=10`` with 1000
            training steps runs 100 inference steps. Only meaningful with
            DDIMScheduler; ignored (``None``) for DDPM.
        renorm (callable, optional): Applied to the output tensor to convert
            images back to their original range (e.g. inverse of the
            normalization used at training time).
        device (torch.device, optional): Target device. Defaults to the model's
            current device.
        generator (torch.Generator, optional): RNG for reproducibility.

    Returns:
        torch.Tensor: Generated images of shape ``(batch_size, C, H, W)``.
            Range is ``[-1, 1]`` if ``renorm`` is ``None``, otherwise whatever
            ``renorm`` maps to.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    num_train_timesteps = noise_scheduler.config.num_train_timesteps
    num_inference_steps = num_train_timesteps // ddim_thinning if ddim_thinning is not None else num_train_timesteps
    noise_scheduler.set_timesteps(num_inference_steps)

    images = torch.randn(
        (batch_size, *image_shape),
        device=device,
        generator=generator,
    )

    if labels is not None:
        labels = torch.as_tensor(labels, dtype=torch.long, device=device)

    for t in tqdm(noise_scheduler.timesteps, desc="Sampling"):
        timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)

        if labels is not None:
            pred = model(
                images,
                timestep=timesteps,
                class_labels=labels,
                return_dict=False,
            )[0]
        else:
            pred = model(images, timesteps, return_dict=False)[0]

        images = noise_scheduler.step(pred, t, images, generator=generator).prev_sample

    if renorm is not None:
        images = renorm(images)

    return images


class PCAEncoder(torch.nn.Module):
    """Linear PCA encoder as an nn.Module.

    Projects ``(N, C, H, W)`` images onto the top-k principal components,
    returning ``(N, rank)`` feature vectors.
    """

    def __init__(self, V: torch.Tensor, center: torch.Tensor):
        super().__init__()
        self.register_buffer("V", V)
        self.register_buffer("center", center)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Args:
            images: ``(N, C, H, W)`` float64 tensor.
        """
        return (images.flatten(1) - self.center) @ self.V


def build_pca_encoder(train_images: torch.Tensor, rank: int = 16) -> PCAEncoder:
    """Build a PCAEncoder from training images via low-rank SVD.

    A lightweight alternative to a learned encoder for use with
    ``compute_fid`` / ``compute_kid`` when no domain-specific encoder is
    available.

    Args:
        train_images: ``(N, C, H, W)`` float64 tensor of training images.
        rank: Number of principal components to retain.

    Returns:
        PCAEncoder mapping ``(N, C, H, W)`` images to ``(N, rank)`` features.
    """
    X = train_images.flatten(1)
    center = X.mean(0)
    _, _, Vt = torch.linalg.svd(X - center, full_matrices=False)
    V = Vt[:rank].T  # (D, rank)
    return PCAEncoder(V, center)


def _sqrtm_sym(A: torch.Tensor) -> torch.Tensor:
    """Matrix square root for a symmetric PSD matrix via eigendecomposition."""
    L, V = torch.linalg.eigh(A)
    return V @ torch.diag(L.clamp(min=0).sqrt()) @ V.mT


def compute_fid(feats_real: torch.Tensor, feats_fake: torch.Tensor) -> float:
    """Fréchet Inception Distance from pre-computed feature vectors.

    Args:
        feats_real: ``(N, d)`` float64 feature tensor for real samples.
        feats_fake: ``(M, d)`` float64 feature tensor for generated samples.

    Returns:
        Scalar FID value.
    """
    mu_r, mu_g = feats_real.mean(0), feats_fake.mean(0)
    r, g = feats_real, feats_fake
    sigma_r = (r - mu_r).T @ (r - mu_r) / (len(r) - 1)
    sigma_g = (g - mu_g).T @ (g - mu_g) / (len(g) - 1)

    sqrt_sigma_r = _sqrtm_sym(sigma_r)
    M = sqrt_sigma_r @ sigma_g @ sqrt_sigma_r
    trace_covmean = torch.linalg.eigvalsh(M).clamp(min=0).sqrt().sum()

    diff = mu_r - mu_g
    fid = diff @ diff + torch.trace(sigma_r) + torch.trace(sigma_g) - 2 * trace_covmean
    return fid.item()


def compute_kid(
    feats_real: torch.Tensor,
    feats_fake: torch.Tensor,
    degree: int = 3,
    gamma: Optional[float] = None,
    coef: float = 1.0,
    subset_size: int = 1000,
    n_subsets: int = 10,
) -> tuple[float, float]:
    """Kernel Inception Distance (polynomial MMD) from pre-computed features.

    Preferred over FID when sample counts are small (~2k), as the estimator is
    unbiased and has lower variance than FID's covariance-based approach.

    Args:
        feats_real: ``(N, d)`` float64 feature tensor for real samples.
        feats_fake: ``(M, d)`` float64 feature tensor for generated samples.
        degree: Polynomial kernel degree. Defaults to ``3``.
        gamma: Kernel scale; defaults to ``1/d``.
        coef: Kernel offset. Defaults to ``1.0``.
        subset_size: Samples per subset for the MMD estimate.
        n_subsets: Number of random subsets to average over.

    Returns:
        ``(mean_kid, std_kid)`` across subsets.
    """
    _gamma = 1.0 / feats_real.shape[1] if gamma is None else gamma

    def poly_kernel(x, y):
        return (_gamma * (x @ y.mT) + coef) ** degree

    scores = []
    for _ in range(n_subsets):
        ri = feats_real[torch.randperm(len(feats_real), device=feats_real.device)[:subset_size]]
        gi = feats_fake[torch.randperm(len(feats_fake), device=feats_fake.device)[:subset_size]]
        n = subset_size

        kxx = poly_kernel(ri, ri)
        kyy = poly_kernel(gi, gi)
        kxy = poly_kernel(ri, gi)

        mmd2 = (
            (kxx.sum() - kxx.trace()) / (n * (n - 1))
            + (kyy.sum() - kyy.trace()) / (n * (n - 1))
            - 2 * kxy.mean()
        )
        scores.append(mmd2.item())

    scores_t = torch.tensor(scores)
    return float(scores_t.mean()), float(scores_t.std())
