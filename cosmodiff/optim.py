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
    cfg_dropout: float = 0.0,
    conditioning: str = 'discrete',
    ema_sigma_rels: Optional[tuple] = None,
    ema_update_every: int = 1,
    ema_burn_in: int = 0,
    min_snr_gamma: Optional[float] = None,
    sigma_log_normal: Optional[tuple] = None,
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
        cfg_dropout (float): Fraction of training batches where labels are
            replaced with a null token (``class_labels`` mode) or zeros
            (``encoder_hidden_states`` mode) for classifier-free guidance
            training. Set to ``0.0`` (default) to disable CFG.
        conditioning (str): How labels are passed to the model.
            ``'discrete'`` (default) passes integer labels via
            ``class_labels=`` (e.g. ``UNet2DModel`` with ``num_class_embeds``
            or ``DiTTransformer2DModel``).  ``'continuous'`` passes float
            labels via ``encoder_hidden_states=`` (e.g.
            ``UNet2DConditionModel``); labels of shape ``(B, D)`` are
            automatically unsqueezed to ``(B, 1, D)``.
        ema_sigma_rels (tuple of float, optional): If set, enables post-hoc
            EMA tracking via ``ema-pytorch``.  Two values are required (e.g.
            ``(0.03, 0.15)``); they control the two power-function EMA
            profiles checkpointed at each epoch.  Choose them to bracket the
            target ``sigma_rel`` range you want to synthesize, with the lower
            value anchoring your floor and the upper value giving some
            headroom above your ceiling.  After training,
            ``ema.synthesize_ema_model(sigma_rel=...)`` reconstructs any
            target profile from those snapshots.  ``None`` (default) disables
            EMA entirely.
        ema_update_every (int): How often (in optimizer steps) to update the
            EMA profiles.  Defaults to ``1`` (every step), matching the Karras
            et al. 2024 reference implementation and the broader diffusion
            training literature.  Larger values silently distort the EMA
            profile because the time-varying beta formula assumes per-step
            updates — only increase if profiling shows the EMA lerp is a
            meaningful fraction of step time.
        ema_burn_in (int): Number of initial optimizer steps during which the
            EMA is *not* updated.  Defaults to ``0`` (start tracking from step
            0).  When set, the EMA's internal time origin shifts to step
            ``ema_burn_in``, so ``sigma_rel`` is interpreted as a fraction of
            the post-burn-in trajectory ``[ema_burn_in, T_total]``.  Useful to
            avoid contaminating the EMA with the very-early random/unstable
            weights, particularly for wider EMA profiles.
        min_snr_gamma (float, optional): If set, applies Min-SNR loss
            weighting from Hang et al. 2023.  Recommended value is ``5.0``.
            The per-sample loss is multiplied by ``min(SNR(t), gamma)``
            divided by a parameterization-dependent factor (``SNR(t)`` for
            epsilon-prediction, ``SNR(t)+1`` for v-prediction, ``1`` for
            sample-prediction), which balances training signal across
            timesteps and is especially helpful for v-prediction.  ``None``
            (default) disables Min-SNR (uniform MSE).
        sigma_log_normal (tuple of float, optional): If set to a
            ``(P_mean, P_std)`` tuple, samples log-σ from ``Normal(P_mean,
            P_std)`` each step (Karras et al. EDM 2022) instead of uniform
            timestep sampling.  Sampled log-σ is mapped to the nearest
            discrete timestep of the underlying scheduler via
            ``torch.searchsorted``.  EDM defaults are ``(-1.2, 1.2)``.
            ``None`` (default) keeps standard uniform timestep sampling.
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

    # post-hoc EMA (fresh each run; not restored on resume)
    ema = None
    if ema_sigma_rels is not None:
        from pathlib import Path
        from ema_pytorch import PostHocEMA
        ema = PostHocEMA(
            accelerator.unwrap_model(model),
            sigma_rels=ema_sigma_rels,
            checkpoint_every_num_steps='manual',
            checkpoint_folder=Path(output_dir) / 'ema',
            update_every=ema_update_every,
        )

    # null token for CFG label dropout (class_labels mode only)
    cfg_null_token = None
    if cfg_dropout > 0.0 and conditioning == 'discrete':
        cfg_null_token = accelerator.unwrap_model(model).config.num_class_embeds - 1

    # precompute log-σ schedule for EDM-style log-normal sigma sampling
    log_sigma_schedule = None
    if sigma_log_normal is not None:
        alphas_cumprod = noise_scheduler.alphas_cumprod.clamp(min=1e-10)
        log_sigma_schedule = 0.5 * torch.log((1.0 - alphas_cumprod) / alphas_cumprod)
        # monotonically increasing in t; we'll map sampled log-σ to nearest t via searchsorted

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
            if labels is not None and cfg_dropout > 0.0:
                drop_mask = torch.rand(labels.shape[0], device=labels.device) < cfg_dropout
                if conditioning == 'discrete':
                    labels = torch.where(drop_mask, torch.full_like(labels, cfg_null_token), labels)
                else:
                    mask = drop_mask.view(-1, *([1] * (labels.ndim - 1)))
                    labels = torch.where(mask, torch.zeros_like(labels), labels)

            t0 = time.time()

            with accelerator.accumulate(model):
                # --- forward diffusion ----------------------------------
                noise = torch.randn_like(images)
                if sigma_log_normal is not None:
                    # EDM-style log-normal sigma sampling: log σ ~ N(P_mean, P_std)
                    P_mean, P_std = sigma_log_normal
                    log_sigma = P_mean + P_std * torch.randn(images.shape[0], device=images.device)
                    sched = log_sigma_schedule.to(images.device)
                    log_sigma = torch.clamp(log_sigma, sched.min().item(), sched.max().item())
                    timesteps = torch.searchsorted(sched, log_sigma)
                    timesteps = torch.clamp(timesteps, 0, noise_scheduler.config.num_train_timesteps - 1).long()
                else:
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
                        if conditioning == 'discrete':
                            cond_kwargs = {'class_labels': labels}
                        else:
                            cond = labels.float()
                            if cond.ndim == 2:
                                cond = cond.unsqueeze(1)
                            cond_kwargs = {'encoder_hidden_states': cond}
                        pred = model(noisy_images, timestep=timesteps, return_dict=False, **cond_kwargs)[0]
                    else:
                        pred = model(noisy_images, timesteps, return_dict=False)[0]

                    prediction_type = noise_scheduler.config.prediction_type
                    if prediction_type == 'epsilon':
                        target = noise
                    elif prediction_type == 'v_prediction':
                        target = noise_scheduler.get_velocity(images, noise, timesteps)
                    elif prediction_type == 'sample':
                        target = images
                    else:
                        raise ValueError(f"Unsupported prediction_type: {prediction_type}")

                    if min_snr_gamma is None:
                        loss = F.mse_loss(pred, target)
                    else:
                        # Min-SNR loss weighting (Hang et al. 2023). SNR = α_bar / (1 - α_bar).
                        alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)
                        snr = alphas_cumprod[timesteps] / (1.0 - alphas_cumprod[timesteps])
                        clipped_snr = torch.clamp(snr, max=min_snr_gamma)
                        if prediction_type == 'epsilon':
                            weight = clipped_snr / snr
                        elif prediction_type == 'sample':
                            weight = clipped_snr
                        else:  # v_prediction
                            weight = clipped_snr / (snr + 1.0)
                        per_sample_mse = F.mse_loss(pred, target, reduction='none').mean(
                            dim=list(range(1, pred.ndim))
                        )
                        loss = (weight * per_sample_mse).mean()

                # --- backward -------------------------------------------
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if ema is not None and global_step >= ema_burn_in:
                ema.update()

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
                    "ema_sigma_rels": list(ema_sigma_rels) if ema_sigma_rels is not None else None,
                    "ema_burn_in": ema_burn_in,
                }
                with open(os.path.join(ckpt_save_path, "checkpoint_config.yaml"), "w") as f:
                    yaml.dump(_to_yaml_safe(ckpt_cfg), f)

                # Model weights (via hook) + optimizer moments + grad scaler + RNG
                accelerator.save_state(ckpt_save_path)

                if ema is not None and ema.step.item() > 0:
                    from pathlib import Path
                    ema.checkpoint_folder = Path(ckpt_save_path) / 'ema'
                    ema.checkpoint_folder.mkdir(exist_ok=True)
                    ema.checkpoint()

                if hasattr(dataset, "augmentations") and dataset.augmentations is not None:
                    with open(os.path.join(ckpt_save_path, "augmentations.pkl"), "wb") as f:
                        pickle.dump(dataset.augmentations, f)

                metrics_path = os.path.join(ckpt_save_path, "metrics.json")
                utils.write_metrics(metrics, metrics_path)

                if verbose:
                    accelerator.print(f"Checkpoint saved → {ckpt_save_path}")

    accelerator.end_training()
    return {'metrics': metrics, 'ema': ema}


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    noise_scheduler,
    batch_size: int = 1,
    image_shape: tuple[int, ...] = (1, 64, 64),
    labels: Optional[torch.Tensor] = None,
    guidance_scale: Optional[float] = None,
    conditioning: str = 'discrete',
    num_steps: Optional[int] = None,
    renorm: Optional[Callable] = None,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Generate a batch of novel images via reverse diffusion.

    Compatible with any ``diffusers`` scheduler exposing
    ``set_timesteps`` / ``step``. The number of inference steps defaults to
    ``noise_scheduler.config.num_train_timesteps`` and can be reduced via
    ``num_steps`` for faster sampling (DDIM, DPM-Solver, Heun, etc.).

    Args:
        model: Trained diffusers model (UNet2DModel or DiTTransformer2DModel).
        noise_scheduler: Trained scheduler (DDPMScheduler, DDIMScheduler, or
            any compatible scheduler with ``set_timesteps`` and ``step``).
        batch_size (int): Number of images to generate.
        image_shape (tuple): Shape of each image ``(C, H, W)``.
        labels (array-like, optional): Conditioning labels of length
            ``batch_size``.  Integer class indices for ``'class_labels'`` mode;
            float array of shape ``(batch_size, D)`` for
            ``'encoder_hidden_states'`` mode.
        guidance_scale (float, optional): Classifier-free guidance scale.
            ``None`` (default) uses the plain conditional prediction with no
            amplification.  A float in ``[1.0, 15.0]`` runs a double forward
            pass with null labels and controls class adherence:
            ``uncond + scale * (cond - uncond)``.
        conditioning (str): Matches the mode used during training.
            ``'discrete'`` (default) for integer class labels;
            ``'continuous'`` for continuous float labels.
        num_steps (int, optional): Number of inference steps to run. Passed
            directly to ``noise_scheduler.set_timesteps``. Defaults to
            ``noise_scheduler.config.num_train_timesteps`` (full schedule).
            Use a smaller value with a higher-order solver (DDIM, DPM-Solver,
            Heun, etc.) for fast sampling.
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
    num_inference_steps = num_steps if num_steps is not None else num_train_timesteps
    noise_scheduler.set_timesteps(num_inference_steps)

    images = torch.randn(
        (batch_size, *image_shape),
        device=device,
        generator=generator,
    )

    if labels is not None:
        if conditioning == 'discrete':
            labels = torch.as_tensor(labels, dtype=torch.long, device=device)
        else:
            labels = torch.as_tensor(labels, dtype=torch.float, device=device)
            if labels.ndim == 2:
                labels = labels.unsqueeze(1)  # (B, D) -> (B, 1, D)

    null_labels = None
    if labels is not None and guidance_scale is not None:
        if conditioning == 'discrete':
            null_token = model.config.num_class_embeds - 1
            null_labels = torch.full_like(labels, null_token)
        else:
            null_labels = torch.zeros_like(labels)

    for t in tqdm(noise_scheduler.timesteps, desc="Sampling"):
        timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)

        if labels is not None:
            cond_key = 'class_labels' if conditioning == 'discrete' else 'encoder_hidden_states'
            pred = model(images, timestep=timesteps, return_dict=False, **{cond_key: labels})[0]
            if null_labels is not None:
                uncond_pred = model(images, timestep=timesteps, return_dict=False, **{cond_key: null_labels})[0]
                pred = uncond_pred + guidance_scale * (pred - uncond_pred)
        else:
            pred = model(images, timesteps, return_dict=False)[0]

        images = noise_scheduler.step(pred, t, images, generator=generator).prev_sample

    if renorm is not None:
        images = renorm(images)

    return images


def gamma_from_sigma_rel(sigma_rel: float) -> float:
    """Convert a sigma_rel value to the corresponding power-function EMA exponent gamma.

    The power-function EMA assigns weight ``w(x) = (gamma+1) * x^gamma`` to
    training progress ``x = s/t`` in ``[0, 1]``.  ``sigma_rel`` is the
    standard deviation of that distribution and is also equal to the relative
    EMA length (fraction of training): ``sigma_rel=0.1`` means the EMA draws
    its weight from roughly the last 10% of training (Karras et al. 2024,
    Sec. 3.1).

    Args:
        sigma_rel: relative standard deviation of the EMA kernel (e.g. 0.05 or 0.28).

    Returns:
        Corresponding gamma exponent.
    """
    from scipy.optimize import brentq

    def _f(gamma):
        mean = (gamma + 1) / (gamma + 2)
        second_moment = (gamma + 1) / (gamma + 3)
        return np.sqrt(second_moment - mean ** 2) - sigma_rel

    return brentq(_f, 1e-4, 1e4)


def compute_ema_profiles(
    sigma_rels_train: tuple,
    checkpoint_epochs: list,
    total_epochs: int,
    sigma_rel_target: float,
) -> dict:
    """Compute per-checkpoint and synthesized EMA weight profiles.

    Implements the analysis behind Fig. 4 of Karras et al. 2024.  For each
    ``(checkpoint_epoch, sigma_rel)`` pair a basis profile is computed; the
    synthesized profile is the non-negative least-squares combination that
    best approximates the target profile.

    Args:
        sigma_rels_train: sigma_rel values used during training (e.g. ``(0.05, 0.28)``).
        checkpoint_epochs: epochs at which ``ema.checkpoint()`` was called.
        total_epochs: total training length (defines the target profile).
        sigma_rel_target: target sigma_rel to synthesize (e.g. ``0.15``).

    Returns:
        dict with keys:
            ``'epochs'``: ``np.ndarray`` of shape ``(total_epochs,)``
            ``'basis'``: list of ``(checkpoint_epoch, sigma_rel, weights)`` tuples
            ``'target'``: ``np.ndarray`` of shape ``(total_epochs,)``
            ``'synthesized'``: ``np.ndarray`` of shape ``(total_epochs,)``
            ``'coefficients'``: ``np.ndarray`` of shape ``(n_basis,)``
    """
    from scipy.optimize import nnls

    epochs = np.arange(1, total_epochs + 1)
    gammas_train = [gamma_from_sigma_rel(s) for s in sigma_rels_train]
    gamma_target = gamma_from_sigma_rel(sigma_rel_target)

    basis = []
    for t_i in checkpoint_epochs:
        for gamma_j, s_j in zip(gammas_train, sigma_rels_train):
            w = np.where(epochs <= t_i, (epochs / t_i) ** gamma_j, 0.0)
            if w.sum() > 0:
                w = w / w.sum()
            basis.append((t_i, s_j, w))

    w_target = (epochs / total_epochs) ** gamma_target
    w_target = w_target / w_target.sum()

    B = np.column_stack([w for _, _, w in basis])
    coeffs, _ = nnls(B, w_target)
    if coeffs.sum() > 0:
        coeffs = coeffs / coeffs.sum()
    w_synth = B @ coeffs

    return {
        'epochs': epochs,
        'basis': basis,
        'target': w_target,
        'synthesized': w_synth,
        'coefficients': coeffs,
    }


def synthesize_ema_from_checkpoints(
    model: torch.nn.Module,
    output_dir: str,
    sigma_rel_target: float,
    sigma_rels: Optional[tuple] = None,
    up_to_epoch: Optional[int] = None,
) -> torch.nn.Module:
    """Synthesize a post-hoc EMA model by pooling snapshots across checkpoints.

    Gathers all ``.pt`` EMA snapshot files from every ``checkpoint-epoch-*/ema/``
    subdirectory in ``output_dir`` into a single temporary directory (via
    symlinks) and delegates synthesis to ``PostHocEMA``.

    ``sigma_rels`` is read automatically from ``checkpoint_config.yaml`` if not
    supplied explicitly.

    Args:
        model: the nn.Module used during training (needed for parameter structure).
        output_dir: root training directory containing ``checkpoint-epoch-*`` dirs.
        sigma_rel_target: target EMA profile to synthesize, e.g. ``0.15``.
        sigma_rels: sigma_rel values used during training, e.g. ``(0.05, 0.28)``.
            If ``None``, read from the latest checkpoint's ``checkpoint_config.yaml``.
        up_to_epoch: if set, only use checkpoints at or before this epoch number.

    Returns:
        Synthesized nn.Module with EMA weights.
    """
    import tempfile
    from pathlib import Path
    from ema_pytorch import PostHocEMA

    ckpt_dirs = sorted(Path(output_dir).glob('checkpoint-epoch-*'))
    if up_to_epoch is not None:
        ckpt_dirs = [d for d in ckpt_dirs if int(d.name.split('-')[-1]) <= up_to_epoch]

    if not ckpt_dirs:
        raise ValueError(f"No checkpoints found in {output_dir}")

    if sigma_rels is None:
        cfg_path = ckpt_dirs[-1] / 'checkpoint_config.yaml'
        with open(cfg_path) as f:
            ckpt_cfg = yaml.safe_load(f)
        sigma_rels = ckpt_cfg.get('ema_sigma_rels')
        if sigma_rels is None:
            raise ValueError(
                "sigma_rels not found in checkpoint_config.yaml — "
                "pass sigma_rels explicitly or retrain with ema_sigma_rels set."
            )
        sigma_rels = tuple(sigma_rels)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for ckpt_dir in ckpt_dirs:
            for pt_file in (ckpt_dir / 'ema').glob('*.pt'):
                (tmp_path / pt_file.name).symlink_to(pt_file.resolve())

        ema = PostHocEMA(
            model,
            sigma_rels=sigma_rels,
            checkpoint_folder=tmp_path,
            checkpoint_every_num_steps='manual',
        )
        return ema.synthesize_ema_model(sigma_rel=sigma_rel_target)


def load_ema_snapshot(
    model: torch.nn.Module,
    checkpoint_dir: str,
    profile_index: int = 0,
) -> torch.nn.Module:
    """Load a single raw EMA snapshot into ``model`` in-place.

    Reads the ``.pt`` snapshot file for the given training profile from
    ``<checkpoint_dir>/ema/`` and copies its weights into ``model`` (no
    synthesis, no pooling across checkpoints).

    Use this when you want exactly the EMA at one of the trained ``sigma_rels``
    captured at one specific epoch.  For arbitrary target ``sigma_rel`` values
    or for pooling across many checkpoints, use
    :func:`synthesize_ema_from_checkpoints` instead.

    Args:
        model: the nn.Module to load weights into; modified in place and returned.
        checkpoint_dir: a ``checkpoint-epoch-XXXX`` directory containing an
            ``ema/`` subdirectory.
        profile_index: which trained profile to load — ``0`` for
            ``sigma_rels[0]`` (e.g. 0.05), ``1`` for ``sigma_rels[1]``
            (e.g. 0.28).  Defaults to ``0``.

    Returns:
        The same ``model``, with EMA weights loaded.
    """
    from pathlib import Path

    ema_dir = Path(checkpoint_dir) / 'ema'
    pt_files = sorted(ema_dir.glob(f'{profile_index}.*.pt'))
    if not pt_files:
        raise ValueError(
            f"No EMA snapshots for profile {profile_index} in {ema_dir}"
        )
    # filename format is {profile_index}.{global_step}.pt — pick the latest step
    pt_file = max(pt_files, key=lambda p: int(p.stem.split('.')[1]))

    snap = torch.load(pt_file, map_location='cpu', weights_only=False)
    ema_state = {
        k[len('ema_model.'):]: v
        for k, v in snap.items()
        if k.startswith('ema_model.')
    }
    model.load_state_dict(ema_state)
    return model


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
