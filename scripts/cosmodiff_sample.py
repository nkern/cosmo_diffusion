#!/usr/bin/env python
"""cosmodiff sampling script.

Usage:
    python cosmodiff_sample.py --checkpoint path/to/checkpoint --n_samples 100 --output samples.npy
    python cosmodiff_sample.py --output_dir path/to/run --n_samples 64 --image_shape 1 64 64

Fast sampling:
    --scheduler DPMSolverMultistepScheduler --num_steps 25

EMA:
    --ema_sigma_rel 0.05  → synthesize EMA at target sigma_rel (uses all checkpoints in output_dir)
"""

import argparse
import os
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description="Generate samples from a cosmodiff checkpoint.")
    ckpt_group = parser.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument(
        "--checkpoint",
        type=str,
        help="Path to a specific checkpoint directory.",
    )
    ckpt_group.add_argument(
        "--output_dir",
        type=str,
        help="Training output directory; the latest checkpoint is used automatically.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="samples.npy",
        help="Path for the output .npy file. Default: samples.npy",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=64,
        help="Total number of samples to generate.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Samples generated per forward pass. Defaults to n_samples (single batch).",
    )
    parser.add_argument(
        "--image_shape",
        type=int,
        nargs="+",
        default=None,
        help="Image shape as C [D] H W (e.g. --image_shape 1 64 64). "
             "Inferred from model config when not set.",
    )
    parser.add_argument(
        "--labels",
        type=int,
        nargs="+",
        default=None,
        help="Discrete class labels for conditional generation. Must be length 1 "
             "(broadcast to all samples) or n_samples. Use --continuous_labels for "
             "encoder_hidden_states-style conditioning.",
    )
    parser.add_argument(
        "--continuous_labels",
        type=str,
        default=None,
        help="Path to a .npy file of float conditioning vectors of shape "
             "(n_samples, D) or (1, D) (broadcast). Used with --conditioning continuous.",
    )
    parser.add_argument(
        "--conditioning",
        type=str,
        choices=["discrete", "continuous"],
        default="discrete",
        help="Conditioning mode used during training. Defaults to 'discrete'.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="Classifier-free guidance scale at inference. None (default) disables "
             "CFG amplification. Typical values 1.0-7.0.",
    )
    parser.add_argument(
        "--ema_sigma_rel",
        type=float,
        default=None,
        help="Target sigma_rel for post-hoc EMA weight synthesis. When set, the "
             "model weights are replaced by the synthesized EMA (across all "
             "checkpoints in output_dir) before sampling.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default=None,
        help="Name of a diffusers scheduler class to use at inference, e.g. "
             "'DDIMScheduler', 'HeunDiscreteScheduler', 'DPMSolverMultistepScheduler'. "
             "Defaults to the training scheduler. The new scheduler is built via "
             ".from_config() to inherit the trained beta schedule and prediction_type.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="Number of inference steps. Defaults to the scheduler's "
             "num_train_timesteps (full schedule).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override, e.g. 'cpu' or 'cuda:0'. "
             "Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    from cosmodiff.utils import load_checkpoint, find_latest_checkpoint
    from cosmodiff.optim import generate, synthesize_ema_from_checkpoints

    # --- resolve checkpoint ---------------------------------------------
    if args.checkpoint is not None:
        ckpt_path = args.checkpoint
        output_dir = os.path.dirname(os.path.abspath(ckpt_path))
    else:
        output_dir = args.output_dir
        ckpt_path = find_latest_checkpoint(output_dir)
        if ckpt_path is None:
            raise FileNotFoundError(f"No checkpoints found in {output_dir}")

    if args.verbose:
        print(f"Loading checkpoint: {ckpt_path}")

    model, noise_scheduler, _optimizer, _lr_scheduler, _augmentations = load_checkpoint(ckpt_path)

    # --- swap to a different scheduler if requested ---------------------
    if args.scheduler is not None:
        import diffusers
        sched_cls = getattr(diffusers, args.scheduler)
        noise_scheduler = sched_cls.from_config(noise_scheduler.config)
        if args.verbose:
            print(f"Switched to {args.scheduler} for inference.")

    # --- device ---------------------------------------------------------
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    if args.verbose:
        print(f"Using device: {device}")

    # --- EMA synthesis (overrides model weights) ------------------------
    if args.ema_sigma_rel is not None:
        if args.verbose:
            print(f"Synthesizing EMA weights at sigma_rel={args.ema_sigma_rel} from {output_dir}")
        model = synthesize_ema_from_checkpoints(
            model, output_dir, sigma_rel_target=args.ema_sigma_rel,
        )
        model = model.to(device)
        model.eval()

    # --- image shape ----------------------------------------------------
    if args.image_shape is not None:
        image_shape = tuple(args.image_shape)
    else:
        cfg = model.config
        # try common diffusers config keys
        in_channels = getattr(cfg, "in_channels", 1)
        sample_size = getattr(cfg, "sample_size", 64)
        if isinstance(sample_size, (list, tuple)):
            image_shape = (in_channels, *sample_size)
        else:
            image_shape = (in_channels, sample_size, sample_size)
        if args.verbose:
            print(f"Inferred image_shape: {image_shape}")

    # --- labels ---------------------------------------------------------
    labels_tensor = None
    if args.conditioning == "discrete":
        if args.labels is not None:
            if len(args.labels) == 1:
                labels_tensor = torch.full((args.n_samples,), args.labels[0], dtype=torch.long)
            elif len(args.labels) == args.n_samples:
                labels_tensor = torch.tensor(args.labels, dtype=torch.long)
            else:
                raise ValueError(
                    f"--labels must be length 1 or n_samples ({args.n_samples}), "
                    f"got {len(args.labels)}."
                )
    else:  # continuous
        if args.continuous_labels is None:
            raise ValueError(
                "--continuous_labels is required when --conditioning continuous."
            )
        cont_arr = np.load(args.continuous_labels)
        if cont_arr.ndim != 2:
            raise ValueError(
                f"--continuous_labels must be 2D (N, D); got shape {cont_arr.shape}."
            )
        if cont_arr.shape[0] == 1:
            cont_arr = np.broadcast_to(cont_arr, (args.n_samples, cont_arr.shape[1]))
        elif cont_arr.shape[0] != args.n_samples:
            raise ValueError(
                f"--continuous_labels first dim must be 1 or n_samples ({args.n_samples}), "
                f"got {cont_arr.shape[0]}."
            )
        labels_tensor = torch.as_tensor(np.asarray(cont_arr), dtype=torch.float32)

    # --- generator ------------------------------------------------------
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    # --- generate in batches --------------------------------------------
    batch_size = args.batch_size if args.batch_size is not None else args.n_samples
    all_samples = []
    remaining = args.n_samples
    offset = 0

    while remaining > 0:
        bs = min(batch_size, remaining)
        batch_labels = None
        if labels_tensor is not None:
            batch_labels = labels_tensor[offset: offset + bs]

        samples = generate(
            model=model,
            noise_scheduler=noise_scheduler,
            batch_size=bs,
            image_shape=image_shape,
            labels=batch_labels,
            guidance_scale=args.guidance_scale,
            conditioning=args.conditioning,
            num_steps=args.num_steps,
            device=device,
            generator=generator,
        )
        all_samples.append(samples.cpu().numpy())
        offset += bs
        remaining -= bs
        if args.verbose:
            print(f"Generated {offset}/{args.n_samples} samples")

    result = np.concatenate(all_samples, axis=0)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(args.output, result)
    print(f"Saved {result.shape} array to {args.output}")


if __name__ == "__main__":
    main()
