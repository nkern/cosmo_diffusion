# cosmo_diffusion

Train 2D/3D diffusion models (UNet and DiT) for cosmological applications.

## Install
```bash
git clone https://github.com/nkern/cosmo_diffusion
cd cosmo_diffusion
pip install .
```

## Running
Look at `cosmodiff/configs/config.yaml` for a configuration file. Then just run:

```bash
cosmodiff_train.py --config path_to_config
```

checkpointing and metrics are automatically stored in `output_dir` (defined in the config).
