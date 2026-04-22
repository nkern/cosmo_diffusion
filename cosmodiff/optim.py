import os
import pickle
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
    dataloader_num_workers: int = 0,
    max_grad_norm: float = 1.0,
    force_cpu: bool = False,
    pin_memory: bool = False,
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
    accelerator.init_trackers(project_name="cosmodiff")

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
            noise_pred = model(
                images,
                timestep=timesteps,
                class_labels=labels,
                return_dict=False,
            )[0]
        else:
            noise_pred = model(images, timesteps, return_dict=False)[0]

        step_kwargs = {}
        if "generator" in noise_scheduler.step.__code__.co_varnames:
            step_kwargs["generator"] = generator
        images = noise_scheduler.step(noise_pred, t, images, **step_kwargs).prev_sample

    if renorm is not None:
        images = renorm(images)

    return images
