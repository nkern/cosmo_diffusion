#!/usr/bin/env python
"""cosmodiff sampling script.

Usage:
    python cosmodiff_sample.py --checkpoint path/to/checkpoint --n_samples 100 --output samples.npy
    python cosmodiff_sample.py --output_dir path/to/run --n_samples 64 --image_shape 1 64 64
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
        help="Class labels for conditional generation. Must be length 1 "
             "(broadcast to all samples) or n_samples.",
    )
    parser.add_argument(
        "--ddim_thinning",
        type=int,
        default=None,
        help="Thinning factor to reduce inference steps (e.g. 10 → 100 steps from 1000). "
             "Automatically switches to DDIMScheduler regardless of what was used during training.",
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
    from cosmodiff.optim import generate

    # --- resolve checkpoint ---------------------------------------------
    if args.checkpoint is not None:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = find_latest_checkpoint(args.output_dir)
        if ckpt_path is None:
            raise FileNotFoundError(f"No checkpoints found in {args.output_dir}")

    if args.verbose:
        print(f"Loading checkpoint: {ckpt_path}")

    model, noise_scheduler, _optimizer, _lr_scheduler, _augmentations = load_checkpoint(ckpt_path)

    # --- swap to DDIM if thinning requested -----------------------------
    if args.ddim_thinning is not None:
        from diffusers import DDIMScheduler
        noise_scheduler = DDIMScheduler.from_config(noise_scheduler.config)
        if args.verbose:
            print("Switched to DDIMScheduler for fast sampling.")

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
            ddim_thinning=args.ddim_thinning,
            device=device,
            generator=generator,
        )
        all_samples.append(samples.cpu().numpy())
        offset += bs
        remaining -= bs
        if args.verbose:
            print(f"Generated {offset}/{args.n_samples} samples")

    result = np.concatenate(all_samples, axis=0)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.save(args.output, result)
    print(f"Saved {result.shape} array to {args.output}")


if __name__ == "__main__":
    main()
