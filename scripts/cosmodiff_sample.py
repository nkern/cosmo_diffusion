#!/usr/bin/env python
"""cosmodiff sampling script.

Usage:
    python cosmodiff_sample.py --config config.yaml
    python cosmodiff_sample.py --checkpoints_dir path/to/run --n_samples 100 --filepath samples.npz
    python cosmodiff_sample.py --checkpoints_dir path/to/run --checkpoint_epoch 50

Config-driven (generate: section sets defaults, CLI args supersede):
    python cosmodiff_sample.py --config config.yaml --n_samples 128

Fast sampling (diffusion-trained models):
    --scheduler DPMSolverMultistepScheduler --num_steps 25

Fast sampling (flow-matching-trained models):
    --scheduler FlowMatchHeunDiscreteScheduler --num_steps 25

EMA:
    --ema_sigma_rel 0.05  → synthesize EMA at target sigma_rel (uses all checkpoints in checkpoints_dir)
"""

import argparse
import os
import numpy as np
import torch
import yaml


_ARRAY_EXTS = (".npy", ".txt", ".csv")


def _load_array_file(path, dtype):
    """Load a 1D/2D array from a .npy, .txt, or .csv file."""
    if path.endswith(".npy"):
        return np.load(path).astype(dtype)
    delim = "," if path.endswith(".csv") else None
    return np.loadtxt(path, dtype=dtype, delimiter=delim)


def _resolve_int_labels(value):
    """value is a list of strs (CLI) or list/str (config): a single filepath
    (.npy/.txt/.csv) loads an int array; otherwise entries are parsed as ints."""
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], str) \
            and value[0].endswith(_ARRAY_EXTS):
        value = value[0]
    if isinstance(value, str):
        return _load_array_file(value, np.int64).ravel()
    return np.array([int(x) for x in value], dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="Generate samples from a cosmodiff checkpoint.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a cosmodiff yaml config file. The generate: section provides defaults "
             "that are superseded by any explicitly passed CLI arguments. "
             "io.output_dir is used as --checkpoints_dir when the latter is not given.",
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default=None,
        help="Directory holding the training checkpoints (the output_dir of the "
             "training run). The latest checkpoint is used when --checkpoint_epoch "
             "is not provided.",
    )
    parser.add_argument(
        "--checkpoint_epoch",
        type=str,
        default=None,
        help="Specific checkpoint epoch to load from --checkpoints_dir, e.g. 50 "
             "→ '<checkpoints_dir>/checkpoint-epoch-0050'. When omitted, the "
             "latest checkpoint in --checkpoints_dir is used.",
    )
    parser.add_argument(
        "--filepath",
        type=str,
        default="samples.npz",
        help="Path for the output .npz file. Default: samples.npz",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
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
        type=str,
        nargs="+",
        default=None,
        help="Discrete class labels for conditional generation. Either a list of "
             "ints (length 1 broadcasts; length n_samples used as-is) or a single "
             "path to a .npy/.txt/.csv file of integer labels. Use --continuous_labels "
             "for encoder_hidden_states-style conditioning.",
    )
    parser.add_argument(
        "--continuous_labels",
        type=str,
        default=None,
        help="Path to a .npy/.txt/.csv file of float conditioning vectors of shape "
             "(n_samples, D) or (1, D) (broadcast). Used with --conditioning continuous.",
    )
    parser.add_argument(
        "--conditioning",
        type=str,
        choices=["discrete", "continuous"],
        default=None,
        help="Conditioning mode used during training. Default: 'discrete'.",
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
             "checkpoints in checkpoints_dir) before sampling.",
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
        "--s_churn",
        type=float,
        default=None,
        help="EDM-style stochasticity injection (Karras et al. 2022). "
             "0 = pure ODE; larger = more SDE-like. Only consumed by "
             "Euler/Heun-family schedulers (diffusion or FM); silently "
             "ignored for others (DDPM, DDIM, DPM-Solver, etc.).",
    )
    parser.add_argument(
        "--s_tmin",
        type=float,
        default=None,
        help="Lower-bound timestep for churn gating; ignored when --s_churn is unset.",
    )
    parser.add_argument(
        "--s_tmax",
        type=float,
        default=None,
        help="Upper-bound timestep for churn gating; ignored when --s_churn is unset.",
    )
    parser.add_argument(
        "--s_noise",
        type=float,
        default=None,
        help="Multiplier on injected noise magnitude during churn (default 1.0).",
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
        default=None,
    )
    args = parser.parse_args()

    # --- load config defaults -------------------------------------------
    gen_cfg = {}
    io_cfg = {}
    if args.config is not None:
        with open(args.config) as f:
            full_cfg = yaml.safe_load(f)
        gen_cfg = full_cfg.get("generate", {}) or {}
        io_cfg = full_cfg.get("io", {}) or {}

    def cfg(key, fallback=None):
        """Return CLI value if given, else config generate: value, else fallback."""
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            return cli_val
        cfg_val = gen_cfg.get(key)
        if cfg_val is not None:
            return cfg_val
        return fallback

    from cosmodiff.utils import load_checkpoint, find_latest_checkpoint
    from cosmodiff.optim import generate, synthesize_ema_from_checkpoints

    # --- resolve checkpoint ---------------------------------------------
    checkpoints_dir = args.checkpoints_dir or io_cfg.get("output_dir")
    if not checkpoints_dir:
        parser.error("Provide --checkpoints_dir or a --config with io.output_dir.")

    if args.checkpoint_epoch is not None:
        ckpt_path = os.path.join(
            checkpoints_dir, f"checkpoint-epoch-{int(args.checkpoint_epoch):04d}"
        )
        if not os.path.isdir(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    else:
        ckpt_path = find_latest_checkpoint(checkpoints_dir)
        if ckpt_path is None:
            raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")

    verbose = cfg("verbose", fallback=False)

    if verbose:
        print(f"Loading checkpoint: {ckpt_path}")

    model, noise_scheduler, _optimizer, _lr_scheduler, _augmentations = load_checkpoint(ckpt_path)

    # --- swap to a different scheduler if requested ---------------------
    scheduler = cfg("scheduler")
    if scheduler is not None:
        import diffusers
        sched_cls = getattr(diffusers, scheduler)
        noise_scheduler = sched_cls.from_config(noise_scheduler.config)
        if verbose:
            print(f"Switched to {scheduler} for inference.")

    # --- device ---------------------------------------------------------
    device_str = cfg("device")
    if device_str is not None:
        device = torch.device(device_str)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    if verbose:
        print(f"Using device: {device}")

    # --- EMA synthesis (overrides model weights) ------------------------
    ema_sigma_rel = cfg("ema_sigma_rel")
    if ema_sigma_rel is not None:
        if verbose:
            print(f"Synthesizing EMA weights at sigma_rel={ema_sigma_rel} from {checkpoints_dir}")
        model = synthesize_ema_from_checkpoints(
            model, checkpoints_dir, sigma_rel_target=ema_sigma_rel,
        )
        model = model.to(device)
        model.eval()

    # --- image shape ----------------------------------------------------
    image_shape_raw = cfg("image_shape")
    if image_shape_raw is not None:
        image_shape = tuple(image_shape_raw)
    else:
        model_cfg = model.config
        in_channels = getattr(model_cfg, "in_channels", 1)
        sample_size = getattr(model_cfg, "sample_size", 64)
        if isinstance(sample_size, (list, tuple)):
            image_shape = (in_channels, *sample_size)
        else:
            image_shape = (in_channels, sample_size, sample_size)
        if verbose:
            print(f"Inferred image_shape: {image_shape}")

    # --- labels ---------------------------------------------------------
    conditioning = cfg("conditioning", fallback="discrete")
    n_samples = cfg("n_samples", fallback=64)

    labels_tensor = None
    if conditioning == "discrete":
        labels = cfg("labels")
        if labels is not None:
            labels_arr = _resolve_int_labels(labels)
            if labels_arr.size == 1:
                labels_tensor = torch.full((n_samples,), int(labels_arr[0]), dtype=torch.long)
            elif labels_arr.size == n_samples:
                labels_tensor = torch.as_tensor(labels_arr, dtype=torch.long)
            else:
                raise ValueError(
                    f"--labels must be length 1 or n_samples ({n_samples}), "
                    f"got {labels_arr.size}."
                )
    else:  # continuous
        continuous_labels = cfg("continuous_labels")
        if continuous_labels is None:
            raise ValueError(
                "--continuous_labels is required when --conditioning continuous."
            )
        cont_arr = np.atleast_2d(_load_array_file(continuous_labels, np.float32))
        if cont_arr.ndim != 2:
            raise ValueError(
                f"--continuous_labels must be 2D (N, D); got shape {cont_arr.shape}."
            )
        if cont_arr.shape[0] == 1:
            cont_arr = np.broadcast_to(cont_arr, (n_samples, cont_arr.shape[1]))
        elif cont_arr.shape[0] != n_samples:
            raise ValueError(
                f"--continuous_labels first dim must be 1 or n_samples ({n_samples}), "
                f"got {cont_arr.shape[0]}."
            )
        labels_tensor = torch.as_tensor(np.asarray(cont_arr), dtype=torch.float32)

    # --- generator ------------------------------------------------------
    seed = cfg("seed")
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    # --- generate in batches --------------------------------------------
    batch_size = cfg("batch_size") or n_samples
    all_samples = []
    remaining = n_samples
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
            guidance_scale=cfg("guidance_scale"),
            conditioning=conditioning,
            num_steps=cfg("num_steps"),
            s_churn=cfg("s_churn"),
            s_tmin=cfg("s_tmin"),
            s_tmax=cfg("s_tmax"),
            s_noise=cfg("s_noise"),
            device=device,
            generator=generator,
        )
        all_samples.append(samples.cpu().numpy())
        offset += bs
        remaining -= bs
        if verbose:
            print(f"Generated {offset}/{n_samples} samples")

    result = np.concatenate(all_samples, axis=0)

    out_dir = os.path.dirname(os.path.abspath(args.filepath))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    save_kwargs = {"samples": result, "ckpt_path": np.array(ckpt_path)}
    if args.config is not None:
        save_kwargs["config"] = np.array(full_cfg, dtype=object)
    np.savez(args.filepath, **save_kwargs)
    print(f"Saved {result.shape} array to {args.filepath}")


if __name__ == "__main__":
    main()
