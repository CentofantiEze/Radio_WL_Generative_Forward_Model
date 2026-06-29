"""Plotting helpers for the radio weak-lensing sampling pipeline.

Each function corresponds to exactly one PNG saved by ``scripts/run.py`` —
the call site in ``run.py`` shrinks to a single line under its
``if args.save_plots:`` gate.

Conventions:
- All functions accept ``out_dir`` (``str`` or ``pathlib.Path``) and write
  their PNG inside it. The directory is assumed to exist already.
- Each function closes its figure on exit so repeated calls don't accumulate
  memory.
- The callers (currently ``run.py``) gate these functions with their own
  ``if args.save_plots:`` check — keeping the gate out of the plotting
  module makes the functions testable in isolation.
- Where many ``args`` fields are needed, the function takes ``args`` (or a
  Namespace-like object) directly. Where only one or two scalars are
  needed, they are passed explicitly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import corner
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange

from .func_utils import stack_2_complex, to_unit_disk
from .posterior_utils import plot_gmm_contours


# --------------------------------------------------------------------------- #
# 1. UV mask + radio PSF                                                      #
# --------------------------------------------------------------------------- #
def plot_uv_mask_psf(mask: np.ndarray, psf: np.ndarray, out_dir: str | Path) -> None:
    """Side-by-side UV mask and (dirty) radio PSF."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(np.real(mask))
    axes[0].set_title("UV mask")
    plt.colorbar(axes[0].images[0], ax=axes[0])
    axes[1].imshow(psf)
    axes[1].set_title("Radio PSF")
    plt.colorbar(axes[1].images[0], ax=axes[1])
    fig.savefig(os.path.join(out_dir, "radio_psf.png"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 2. Tiled grid of UV observations                                            #
# --------------------------------------------------------------------------- #
def _build_uv_image_list(
    data: np.ndarray, mask: np.ndarray, uv_pos
) -> list[np.ndarray]:
    """Place visibilities back onto the sparse UV grid (one image per galaxy)."""
    out = []
    for vis in stack_2_complex(data, batch=True):
        img = np.zeros_like(mask)
        img[uv_pos] = vis
        out.append(img)
    return out


def plot_data_grid(
    data: np.ndarray,
    mask: np.ndarray,
    uv_pos,
    Ngal: int,
    out_dir: str | Path,
) -> list[np.ndarray]:
    """Tile the first up-to-100 UV observations into one image.

    Returns the per-galaxy complex UV images so the caller can reuse them for
    other plots (e.g. ``plot_random_galaxy``).
    """
    uv_images = _build_uv_image_list(data, mask, uv_pos)
    if Ngal >= 100:
        tiled = rearrange(
            uv_images[:100], "(n1 n2) h w -> (n1 h) (n2 w)", n1=10, n2=10
        )
    else:
        side = int(np.ceil(np.sqrt(Ngal)))
        tiled = rearrange(
            uv_images[: side * side],
            "(n1 n2) h w -> (n1 h) (n2 w)",
            n1=side,
            n2=side,
        )

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(np.abs(tiled), vmin=np.min(np.abs(tiled)), vmax=np.max(np.abs(tiled)))
    plt.colorbar()
    fig.savefig(os.path.join(out_dir, "radio_data.png"))
    plt.close(fig)
    return uv_images


# --------------------------------------------------------------------------- #
# 3. Random galaxy (UV plane + image plane)                                   #
# --------------------------------------------------------------------------- #
def plot_random_galaxy(
    uv_images: list[np.ndarray],
    Ngal: int,
    out_dir: str | Path,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """Pick one random galaxy from ``uv_images`` and show its UV + image view."""
    if rng is None:
        rng = np.random.default_rng()
    idx = int(rng.integers(0, Ngal))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(np.abs(uv_images[idx]))
    axes[0].set_title(f"Observed galaxy {idx} uv")
    plt.colorbar(axes[0].images[0], ax=axes[0])
    axes[1].imshow(np.abs(np.fft.ifftshift(np.fft.ifft2(uv_images[idx]))))
    axes[1].set_title(f"Observed galaxy {idx} image")
    plt.colorbar(axes[1].images[0], ax=axes[1])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "radio_data_galaxy.png"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 4. MAP convergence: loss + g1/g2 traces                                     #
# --------------------------------------------------------------------------- #
def plot_map_convergence(
    map_losses,
    map_g1_trace,
    map_g2_trace,
    args,
    *,
    has_shear: bool,
    g_rescale: float,
    out_dir: str | Path,
) -> None:
    """MAP loss curve + per-chain g1/g2 traces."""
    n_panels = 3 if has_shear else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]
    total_steps = (args.n_steps_map_freeze_shear if has_shear else 0) + args.n_steps_map
    steps = jnp.arange(total_steps)

    # Loss panel — log scale, shifted to keep values positive
    loss_min = jnp.min(map_losses)
    loss_offset = jnp.where(loss_min < 0, jnp.abs(loss_min) + 1.0, 0.0)
    for c in range(map_losses.shape[0]):
        axes[0].plot(steps, map_losses[c] + loss_offset, alpha=0.7, label=f"chain {c}")
    axes[0].set_yscale("log")
    if has_shear and args.n_steps_map_freeze_shear > 0:
        axes[0].axvline(
            args.n_steps_map_freeze_shear, color="k", ls=":", alpha=0.5,
            label="unfreeze g",
        )
    axes[0].set_xlabel("MAP step")
    axes[0].set_ylabel(
        "Loss (NLL)" if loss_offset == 0 else f"Loss (NLL + {loss_offset:.1f})"
    )
    axes[0].set_title("MAP loss")
    axes[0].legend(fontsize=7)

    if has_shear:
        # Per-chain trace, shape (n_chains, total_steps, 1) -> drop trailing dim
        map_g1_phys = map_g1_trace.squeeze(-1) * g_rescale
        map_g2_phys = map_g2_trace.squeeze(-1) * g_rescale
        for c in range(map_g1_phys.shape[0]):
            axes[1].plot(steps, map_g1_phys[c], alpha=0.7, label=f"chain {c}")
            axes[2].plot(steps, map_g2_phys[c], alpha=0.7, label=f"chain {c}")
        axes[1].axhline(args.g1_true, color="r", ls="--", lw=1, label="true")
        axes[2].axhline(args.g2_true, color="r", ls="--", lw=1, label="true")
        if args.n_steps_map_freeze_shear > 0:
            axes[1].axvline(args.n_steps_map_freeze_shear, color="k", ls=":", alpha=0.5)
            axes[2].axvline(args.n_steps_map_freeze_shear, color="k", ls=":", alpha=0.5)
        for ax, name in zip((axes[1], axes[2]), ("g1", "g2")):
            ax.set_xlabel("MAP step")
            ax.set_ylabel(name)
            ax.set_title(f"{name} MAP trace")
            ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "map_convergence.png"), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 5. Initial guess / MAP shear vs truth (scatter)                             #
# --------------------------------------------------------------------------- #
def plot_initial_guess_shear(init_val_prior, init_val_map, args, out_dir: str | Path) -> None:
    """Plot the prior-draw and MAP shear estimates in the (g1, g2) plane."""
    g_rescale = args.g_prior_scale / args.g_prior_sigma
    fig = plt.figure()
    plt.scatter(init_val_prior["g1"] * g_rescale,
                init_val_prior["g2"] * g_rescale,
                label="Initial guess")
    plt.scatter(init_val_map["g1"] * g_rescale,
                init_val_map["g2"] * g_rescale,
                label="MAP estimate")
    plt.scatter(args.g1_true, args.g2_true, color="red", label="True shear")
    plt.xlim(args.g1_true - 3 * args.g_prior_scale,
             args.g1_true + 3 * args.g_prior_scale)
    plt.ylim(args.g2_true - 3 * args.g_prior_scale,
             args.g2_true + 3 * args.g_prior_scale)
    plt.xlabel("g1")
    plt.ylabel("g2")
    plt.title("Initial guess for the shear")
    plt.legend()
    fig.savefig(os.path.join(out_dir, "radio_initial_guess.png"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 6. MCMC chains in raw (MCMC-space) coordinates                              #
# --------------------------------------------------------------------------- #
def _chains_figure(labels: Sequence[str]):
    """Create the stacked-panels figure shared by ``plot_chains_raw`` and ``plot_chains_scaled``."""
    n = max(len(labels), 1)
    fig, axes = plt.subplots(n, figsize=(10, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]
    return fig, axes


def plot_chains_raw(
    samples_: dict,
    labels: Sequence[str],
    latent_key: Optional[str],
    num_chains: int,
    plot_chains: str,
    out_dir: str | Path,
) -> None:
    """Raw MCMC-space chain traces. Saves only if ``plot_chains in {'samples', 'both'}``."""
    fig, axes = _chains_figure(labels)
    for i, label in enumerate(labels):
        ax = axes[i]
        for k in range(num_chains):
            if latent_key is not None and label == f"{latent_key}[0]":
                ax.plot(samples_[latent_key][k, :, 0, 0, 0], "k", alpha=0.3)
            else:
                ax.plot(samples_[label][k, :, 0], "k", alpha=0.3)
        ref_key = (
            latent_key
            if (latent_key is not None and label == f"{latent_key}[0]")
            else label
        )
        ax.set_xlim(0, samples_[ref_key].shape[1])
        ax.set_ylabel(label)
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    if plot_chains in ("samples", "both"):
        fig.savefig(os.path.join(out_dir, "radio_chains.png"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 7. MCMC chains in physical units                                            #
# --------------------------------------------------------------------------- #
def _scale_e_pair(samples_: dict, k: int, label: str, args, names: tuple[str, str]):
    """Helper: scale (e1, e2) (or _disk / _bulge variant) to physical, project to unit disk."""
    e = jnp.stack([
        samples_[names[0]][k, :, 0] / args.ell_prior_sigma * args.ell_prior_scale,
        samples_[names[1]][k, :, 0] / args.ell_prior_sigma * args.ell_prior_scale,
    ], 0)
    e = to_unit_disk(e)
    return e[0] if label == names[0] else e[1]


def plot_chains_scaled(
    samples_: dict,
    labels: Sequence[str],
    latent_key: Optional[str],
    args,
    out_dir: str | Path,
) -> None:
    """Chain traces in physical units. Saves only if ``args.plot_chains in {'scaled', 'both'}``."""
    fig, axes = _chains_figure(labels)
    for i, label in enumerate(labels):
        ax = axes[i]
        for k in range(args.num_chains):
            if label in ("hlr", "hlr_disk", "hlr_bulge") and label in samples_:
                ax.plot(
                    jax.nn.sigmoid(samples_[label][k, :, 0] / args.hlr_prior_sigma)
                    * (args.hlr_prior_max - args.hlr_prior_min) + args.hlr_prior_min,
                    "k", alpha=0.3,
                )
            elif label == "flux":
                ax.plot(
                    jax.nn.sigmoid(samples_["flux"][k, :, 0] / args.flux_prior_sigma)
                    * (args.flux_prior_max - args.flux_prior_min) + args.flux_prior_min,
                    "k", alpha=0.3,
                )
            elif label == "flux_ratio":
                ax.plot(
                    jax.nn.sigmoid(samples_["flux_ratio"][k, :, 0])
                    * args.composite_flux_ratio_max,
                    "k", alpha=0.3,
                )
            elif label in ("e1", "e2") and "e1" in samples_ and "e2" in samples_:
                ax.plot(_scale_e_pair(samples_, k, label, args, ("e1", "e2")),
                        "k", alpha=0.3)
            elif label in ("e1_disk", "e2_disk") and "e1_disk" in samples_:
                ax.plot(_scale_e_pair(samples_, k, label, args, ("e1_disk", "e2_disk")),
                        "k", alpha=0.3)
            elif label in ("e1_bulge", "e2_bulge") and "e1_bulge" in samples_:
                ax.plot(_scale_e_pair(samples_, k, label, args, ("e1_bulge", "e2_bulge")),
                        "k", alpha=0.3)
            elif label in ("g1", "g2") and "g1" in samples_ and "g2" in samples_:
                g = jnp.stack([
                    samples_["g1"][k, :, 0] / args.g_prior_sigma * args.g_prior_scale,
                    samples_["g2"][k, :, 0] / args.g_prior_sigma * args.g_prior_scale,
                ], 0)
                g = to_unit_disk(g)
                ax.plot(g[0] if label == "g1" else g[1], "k", alpha=0.3)
            elif latent_key is not None and label == f"{latent_key}[0]":
                ax.plot(samples_[latent_key][k, :, 0, 0, 0] / args.latent_sigma,
                        "k", alpha=0.3)
            else:
                ax.plot(samples_[label][k, :, 0], "k", alpha=0.3)
        ref_key = (
            latent_key
            if (latent_key is not None and label == f"{latent_key}[0]")
            else label
        )
        ax.set_xlim(0, samples_[ref_key].shape[1])
        ax.set_ylabel(label)
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    if args.plot_chains in ("scaled", "both"):
        fig.savefig(os.path.join(out_dir, "radio_chains_scaled.png"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 8. Corner plot of the (g1, g2) posterior                                    #
# --------------------------------------------------------------------------- #
def plot_corner_shear(samples_: dict, args, out_dir: str | Path) -> None:
    """Corner plot of (g1, g2) samples in physical units, with truths overlaid."""
    truths = np.array([args.g1_true, args.g2_true])
    samples_g = np.concatenate([samples_["g1"], samples_["g2"]], -1).reshape((-1, 2)) * (
        args.g_prior_scale / args.g_prior_sigma
    )
    fig = plt.figure(figsize=(7, 7))
    fig = corner.corner(samples_g, truths=truths,
                        labels=[r"$\gamma_1$", r"$\gamma_2$"], fig=fig)
    fig.savefig(os.path.join(out_dir, "radio_corner_g.png"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 9. GMM posterior contours                                                   #
# --------------------------------------------------------------------------- #
def plot_gmm_posterior(gmm_params: dict, args, out_dir: str | Path) -> None:
    """GMM contours of the (g1, g2) posterior with the truth marker."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plot_gmm_contours(gmm_params, ax=ax, true_g=(args.g1_true, args.g2_true))
    ax.set_title("GMM Posterior Density")
    fig.savefig(
        os.path.join(out_dir, "gmm_contours.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
