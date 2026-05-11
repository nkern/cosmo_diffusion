"""Generate a realistic cosmic-web banner for cosmo_diffusion.

Left half: cosmic web realization A.
Right half: a different realization B.
Middle: Gaussian-enveloped noise that smoothly takes over near the centre,
emphasising the diffusion process A → noise → B.
"""

import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr


def power_law_field(shape, alpha=4.5, seed=0):
    """Gaussian random field with power spectrum P(k) ~ k^{-alpha}."""
    h, w = shape
    rng = np.random.default_rng(seed)

    noise = rng.standard_normal((h, w))
    F = np.fft.fft2(noise)

    kx = np.fft.fftfreq(w)
    ky = np.fft.fftfreq(h)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)
    K[0, 0] = 1.0  # avoid div-by-zero

    P = K ** (-alpha)
    P[0, 0] = 0.0
    F_shaped = F * np.sqrt(P)
    field = np.real(np.fft.ifft2(F_shaped))
    return field


def cosmic_web_field(shape, seed):
    """A cosmic-web-ish 2D field: diffuse filamentary background + bright halos."""
    rng = np.random.default_rng(seed)
    h, w = shape

    # Filamentary diffuse component
    field = power_law_field(shape, alpha=2.2, seed=seed)
    field = (field - field.mean()) / field.std()
    field = np.exp(0.75 * field) - 1.0   # log-normal stretch — emphasises peaks
    field = np.clip(field, 0, None)

    # Add bright halos
    #Y, X = np.mgrid[:h, :w]
    #n_halos = int(rng.integers(20, 32))
    #for _ in range(n_halos):
    #    cx = rng.uniform(0, w)
    #    cy = rng.uniform(0, h)
    #    sigma = rng.uniform(3.0, 14.0)
    #    amp = rng.uniform(0.8, 4.0)
    #    field += amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))

    # Normalise to [0, 1] (saturate the brightest tail for punch).  Using a
    # slightly lower percentile boosts midrange brightness so the diffuse
    # filamentary structure remains visible.
    field = field / np.percentile(field, 95.0)
    return np.clip(field, 0.0, 1.0)


def make_banner(out_path, W=1200, H=300):
    print(f"generating {W} x {H} banner …")

    # Two distinct cosmic-web realisations (seeds chosen so the two halves
    # have comparable bright-feature density).
    left = cosmic_web_field((H, W), seed=5)
    right = cosmic_web_field((H, W), seed=4)

    # Noise with cosmic-web-like dynamic range (so the blend looks coherent)
    rng = np.random.default_rng(999)
    noise = rng.standard_normal((H, W))
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = noise ** 1.4
    noise = noise * 1   # slightly dimmer than the bright cosmic peaks

    # Spatial weights along the x-axis
    x = np.arange(W)
    # Gaussian noise envelope — peaks in the middle, falls off toward both edges
    sigma_n = W / 5.0
    noise_env = np.exp(-((x - W / 2) ** 2) / (2 * sigma_n ** 2))

    # Left → right taper using a smooth sigmoid
    transition = W / 9.0
    left_taper = 1.0 / (1.0 + np.exp((x - W / 2) / transition))
    right_taper = 1.0 - left_taper

    left_w = (1.0 - noise_env) * left_taper
    right_w = (1.0 - noise_env) * right_taper
    noise_w = noise_env

    # Compose
    composite = left * left_w[None, :] + right * right_w[None, :] + noise * noise_w[None, :]

    # CMasher's colormap
    cmap = cmr.freeze

    fig, ax = plt.subplots(figsize=(W / 150, H / 150), dpi=150)
    ax.imshow(
        composite,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        interpolation="bicubic",
    )
    # Title at the bottom-center
    ax.text(
        W / 2,
        H - H * 0.06,
        "cosmo_diffusion",
        color="#ffffff",
        fontsize=34,
        fontname="Courier New",
        weight="bold",
        ha="center",
        va="bottom",
        alpha=0.95,
    )
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    print(f"wrote {out_path}")


if __name__ == "__main__":
    make_banner("/Users/nkern/Software/cosmo_diffusion/banner.png")
