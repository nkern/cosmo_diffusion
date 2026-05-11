![banner](docs/_static/banner.png)

Train 2D/3D diffusion (and flow-matching) models — UNet, UNet-Conditional, DiT,
and PixArt — for cosmological applications.

## Install

```bash
git clone https://github.com/nkern/cosmo_diffusion
cd cosmo_diffusion
pip install -e .
```

## Dependencies

- `numpy`
- `torch`
- `diffusers`
- `accelerate`
- `tqdm`
- `pyyaml`
- `scipy`
- `h5py`
- `matplotlib`
- `ema-pytorch`

## Quick demo

Configure a training run in `cosmodiff/data/config.yaml` (paths, model,
scheduler, training kwargs), then launch:

```bash
cosmodiff_train.py --config path/to/config.yaml
```

Checkpoints and metrics are written automatically to the `output_dir` set in
the config.  To sample from a trained checkpoint:

```bash
cosmodiff_sample.py --config path/to/config.yaml \
    --checkpoints_dir /path/to/output \
    --checkpoint_epoch 50 \
    --n_samples 100 \
    --filepath ./samples.npz
```

The `generate:` section of the config supplies the sampling defaults; any flag
passed on the command line overrides the corresponding config value.
`--checkpoints_dir` is the training output directory containing the
`checkpoint-epoch-NNNN/` subdirectories; when omitted, `io.output_dir` from the
config is used.  `--checkpoint_epoch` selects a specific epoch (e.g. 50 →
`checkpoint-epoch-0050`); when omitted, the latest checkpoint is used.  The
output `.npz` always contains `samples` and `ckpt_path`, plus the full config
dictionary when `--config` is provided.

For fast inference, swap in a higher-order solver:

```bash
cosmodiff_sample.py --config path/to/config.yaml \
    --checkpoints_dir /path/to/output \
    --checkpoint_epoch 50 \
    --n_samples 100 \
    --filepath ./samples.npz \
    --scheduler DPMSolverMultistepScheduler \
    --num_steps 25
```

## Authors

- Nicholas Kern
- Jiaming Pan
