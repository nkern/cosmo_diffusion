#!/usr/bin/env python
"""cosmodiff training script.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --num_epochs 100 --batch_size 32
"""

import os
import glob
import argparse
import numpy as np
import torch
import yaml


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

	for k, v in overrides.items():
		if v is not None:
			config["train"][k] = v

	output_dir = config["io"]["output_dir"]

	from cosmodiff.optim import train
	from cosmodiff.utils import load_checkpoint, parse_config_model, parse_config_data

	# --- check for existing checkpoint ----------------------------------
	latest_ckpt = find_latest_checkpoint(output_dir)

	if latest_ckpt is not None:
		print(f"Resuming from checkpoint: {latest_ckpt}")
		model, noise_scheduler, optimizer, lr_scheduler, augmentations = (
			load_checkpoint(latest_ckpt)
		)
		dataset = parse_config_data(config)
		if augmentations is not None:
			dataset.augmentations = augmentations

	else:
		print("No checkpoint found, training from scratch.")
		model, optimizer, noise_scheduler, lr_scheduler = parse_config_model(config)
		dataset = parse_config_data(config)

	metrics = train(
		dataset,
		model,
		optimizer=optimizer,
		noise_scheduler=noise_scheduler,
		lr_scheduler=lr_scheduler,
		output_dir=output_dir,
		**config["train"],
	)

	print(f"Training complete.")
	print(f"Final epoch loss: {metrics['epoch_loss'][-1]:.4f}")
	print(f"Total time: {sum(metrics['epoch_times']):.1f}s")


if __name__ == "__main__":
	main()