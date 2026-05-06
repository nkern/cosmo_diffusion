#!/usr/bin/env python
"""cosmodiff training script.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --num_epochs 100 --batch_size 32
"""

import os
import glob
import shutil
import argparse
import numpy as np
import torch
import yaml


def main():
    parser = argparse.ArgumentParser(description="Train a cosmodiff diffusion model.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a cosmodiff yaml config file.",
    )
    # training overrides
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--mixed_precision", type=str, default=None)
    parser.add_argument("--checkpoint_every_n_epochs", type=int, default=None)
    parser.add_argument("--dataloader_num_workers", type=int, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--no_shuffle", action="store_true")
    parser.add_argument("--verbose", action="store_true", default=None)
    parser.add_argument("--force_cpu", action="store_true", default=None)
    parser.add_argument("--pin_memory", action='store_true', default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # apply CLI overrides to train config
    overrides = {
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "mixed_precision": args.mixed_precision,
        "checkpoint_every_n_epochs": args.checkpoint_every_n_epochs,
        "dataloader_num_workers": args.dataloader_num_workers,
        "max_grad_norm": args.max_grad_norm,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }
    if args.no_shuffle:
        overrides["shuffle"] = False
    if args.verbose:
        overrides["verbose"] = True
    if args.force_cpu:
        overrides["force_cpu"] = True
    if args.pin_memory:
        overrides['pin_memory'] = True

    for k, v in overrides.items():
        if v is not None:
            config["train"][k] = v

    output_dir = config["io"]["output_dir"]

    os.makedirs(output_dir, exist_ok=True)
    config_dest = os.path.join(output_dir, os.path.basename(args.config))
    if not os.path.exists(config_dest):
        shutil.copy2(args.config, config_dest)

    from cosmodiff.optim import train
    from cosmodiff.utils import parse_config_model, parse_config_data, write_metrics, find_latest_checkpoint

    # --- check for existing checkpoint ----------------------------------
    latest_ckpt = find_latest_checkpoint(output_dir)
    dataset, norm = parse_config_data(config)

    if latest_ckpt is not None:
        print(f"Resuming from checkpoint: {latest_ckpt}")
        result = train(
            dataset,
            resume_from_checkpoint=latest_ckpt,
            output_dir=output_dir,
            **config["train"],
        )

    else:
        print("No checkpoint found, training from scratch.")
        model, optimizer, noise_scheduler, lr_scheduler = parse_config_model(config)
        result = train(
            dataset,
            model,
            optimizer=optimizer,
            noise_scheduler=noise_scheduler,
            lr_scheduler=lr_scheduler,
            output_dir=output_dir,
            **config["train"],
        )

    print(f"Training complete.")
    print(f"Final epoch loss: {result['metrics']['epoch_loss'][-1]:.4f}")
    print(f"Total time: {sum(result['metrics']['epoch_times']):.1f}s")

    metrics_path = os.path.join(
        output_dir, "metrics_epoch_{:04d}.json".format(len(result['metrics']["epoch_loss"]) - 1)
    )
    write_metrics(metrics, metrics_path)
    print(f"Metrics written to {metrics_path}")

if __name__ == "__main__":
    main()
