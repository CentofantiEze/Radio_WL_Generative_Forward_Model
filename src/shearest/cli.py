"""Command-line interface for the radio weak-lensing sampling pipeline.

`build_parser()` returns the fully configured `argparse.ArgumentParser`.
`parse_args(argv=None)` is a thin convenience wrapper around it.

Keeping the parser construction here (rather than inline in ``run.py``) lets us:
- inspect / test the CLI without parsing real ``sys.argv``,
- reuse the same parser from notebooks or alternative entry points,
- keep ``run.py`` focused on the pipeline orchestration.
"""

from __future__ import annotations

import argparse
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argparse parser for the sampling pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Radio weak-lensing shear inference: simulates radio interferometric "
            "observations of galaxies and samples the joint posterior over cosmic "
            "shear and per-galaxy nuisance parameters via gradient-based MCMC."
        ),
    )

    # Run parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: set seed randomly).",
    )
    parser.add_argument(
        "--id", type=str, default=None, help="Unique identifier for the run."
    )

    # Scene modelling parameters
    parser.add_argument("--Ngal", type=int, default=100, help="Number of galaxies.")
    parser.add_argument("--Npx", type=int, default=128, help="Image size in pixels.")
    parser.add_argument(
        "--pixel_scale", type=float, default=0.15, help="Pixel scale in arcsec/pixel."
    )
    parser.add_argument(
        "--noise_uv",
        type=float,
        default=0.004,
        help="UV noise level for the model likelihood (and for data generation when --noise_data is not set).",
    )
    parser.add_argument(
        "--noise_data",
        type=float,
        default=None,
        help="UV noise level for data generation. If None, uses --noise_uv (backwards-compatible).",
    )
    parser.add_argument(
        "--trecs_data_path",
        type=str,
        default=None,
        help="Galaxy, hlr and flux fit over the TRECS catalog (trecs_gal_params.npy).",
    )
    parser.add_argument(
        "--deepshape_data_path",
        type=str,
        default=None,
        help="Path to the DeepShape dataset (val_set_rivi.h5).",
    )
    parser.add_argument(
        "--cosmos_data_path",
        type=str,
        default=None,
        help="Path to the COSMOS dataset 23.5 (for real galaxy images).",
    )
    parser.add_argument(
        "--cosmos_sample",
        type=str,
        default="23.5",
        help="COSMOS dataset sample to use: 23.5 or 25.2.",
    )
    parser.add_argument(
        "--mag_cut",
        type=float,
        default=None,
        help="Optional magnitude cut for COSMOS sample.",
    )
    parser.add_argument(
        "--data_profile",
        type=str,
        default="exp",
        help="Galaxy dataset profile type: exp, sersic, spergel, real or VAE.",
    )
    parser.add_argument(
        "--g1_true", type=float, default=-0.05, help="True g1 shear value."
    )
    parser.add_argument(
        "--g2_true", type=float, default=0.05, help="True g2 shear value."
    )
    parser.add_argument(
        "--ell_scale",
        type=float,
        default=0.2,
        help="Ellipticity scale for data generation.",
    )
    parser.add_argument(
        "--sersic_index",
        type=float,
        default=None,
        help="Sersic index or nu if Spergel profil.",
    )
    parser.add_argument(
        "--data_vae_path",
        type=str,
        default=None,
        help="Path to a AE for data generation when --data_profile=VAE. Falls back to --vae_path if not set (must have a trained encoder).",
    )
    parser.add_argument(
        "--data_vae_epoch",
        type=int,
        default=None,
        help="Epoch of the data-generation AE. Falls back to --vae_epoch if not set.",
    )

    # Radio PSF parameters
    parser.add_argument(
        "--antenna_type",
        type=str,
        default="random",
        help="Antenna type: random or file.",
    )
    parser.add_argument(
        "--antenna_file",
        type=str,
        default=None,
        help="Path to antenna file if antenna_type is file.",
    )
    parser.add_argument(
        "--uv_mask_weighting",
        type=str,
        default="binary",
        help="UV weighting: binary or histogram.",
    )
    parser.add_argument("--n_antenna", type=int, default=50, help="Number of antennas.")
    parser.add_argument("--E_lim", type=float, default=50e3, help="East limit.")
    parser.add_argument("--N_lim", type=float, default=50e3, help="North limit.")
    parser.add_argument("--track_time", type=float, default=10, help="Track time.")
    parser.add_argument("--t0", type=float, default=0, help="Start time.")
    parser.add_argument("--n_times", type=int, default=4, help="Number of times.")
    parser.add_argument("--f", type=float, default=1.4e9, help="Frequency.")
    parser.add_argument("--df", type=float, default=None, help="Frequency bandwidth")
    parser.add_argument(
        "--array_lat",
        type=float,
        default=-30,
        help="Latitude of the array in degrees (used for UV mask generation).",
    )
    parser.add_argument(
        "--array_dec",
        type=float,
        default=-30,
        help="Declination of the target field in degrees (used for UV mask generation).",
    )
    parser.add_argument(
        "--n_freqs", type=int, default=1, help="Number of frequency channels."
    )
    parser.add_argument(
        "--radio_array_seed",
        type=int,
        default=123,
        help="Random seed for the radio array generation.",
    )

    # Galaxy generative model parameters
    parser.add_argument(
        "--model_profile",
        type=str,
        default="exp",
        help="Model profile type: exp, spergel or VAE (for realistic galaxy images).",
    )
    parser.add_argument(
        "--ell_prior_sigma",
        type=float,
        default=1.0,
        help="Ellipticity samples range (non-physical).",
    )
    parser.add_argument(
        "--ell_prior_scale", type=float, default=0.2, help="Ellipticity prior scale."
    )
    parser.add_argument(
        "--g_prior_sigma",
        type=float,
        default=1.0,
        help="Shear samples range (non-physical).",
    )
    parser.add_argument(
        "--g_prior_scale", type=float, default=0.1, help="Shear prior scale."
    )
    parser.add_argument(
        "--hlr_prior_sigma",
        type=float,
        default=1.0,
        help="Half-light radius samples range (non-physical).",
    )
    parser.add_argument(
        "--hlr_prior_min", type=float, default=0.1, help="Half-light radius prior min."
    )
    parser.add_argument(
        "--hlr_prior_max", type=float, default=3.0, help="Half-light radius prior max."
    )
    parser.add_argument(
        "--flux_prior_sigma",
        type=float,
        default=1.0,
        help="Flux samples range (non-physical).",
    )
    parser.add_argument(
        "--flux_prior_min", type=float, default=0.03, help="Flux prior min."
    )
    parser.add_argument(
        "--flux_prior_max", type=float, default=0.25, help="Flux prior max."
    )
    parser.add_argument(
        "--composite_flux_ratio_max",
        type=float,
        default=4.0,
        help="Max disk/bulge flux ratio for composite model.",
    )

    # Autoencoder parameters
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=4,
        help="Latent dimension for VAE, z.shape -> (latent_dim, latent_dim).",
    )
    parser.add_argument(
        "--latent_mean",
        type=float,
        default=0.0,
        help="Latent representation mean value.",
    )
    parser.add_argument(
        "--latent_sigma",
        type=float,
        default=1.0,
        help="Latent space sampling sigma.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Path to the trained autoencoder model.",
    )
    parser.add_argument(
        "--vae_epoch",
        type=int,
        default=None,
        help="Epoch of the trained autoencoder model.",
    )
    parser.add_argument(
        "--vae_model_inference_mode",
        type=str,
        default="parallel",
        help="VAE model inference mode: parallel, sequential or batch.",
    )
    parser.add_argument(
        "--vae_inference_batch_size",
        type=int,
        default=1,
        help="VAE inference batch size if using batch mode.",
    )
    parser.add_argument(
        "--use_dropout",
        action="store_true",
        help="Enable VAE dropout during inference (disabled by default for deterministic decoding).",
    )
    parser.add_argument(
        "--vae_precision",
        type=str,
        default="float16",
        choices=["float32", "float16"],
        help="VAE decoder weight precision. float16 gives ~2x speedup on V100 GPU.",
    )
    parser.add_argument(
        "--pixel_scale_vae",
        type=float,
        default=0.03,
        help="Pixel scale for VAE images, default: HST pixel scale (0.03 arcsec/pixel).",
    )

    # Normalizing flow parameters
    parser.add_argument(
        "--use_flow",
        action="store_true",
        help="Enable normalizing flow reparameterization of VAE latent space.",
    )
    parser.add_argument(
        "--flow_path",
        type=str,
        default=None,
        help="Path to the trained flow model directory.",
    )
    parser.add_argument(
        "--flow_epoch",
        type=int,
        default=None,
        help="Epoch of the trained flow model checkpoint.",
    )
    parser.add_argument(
        "--flow_condition",
        type=float,
        nargs="+",
        default=None,
        help="Conditioning vector for the flow model (if applicable).",
    )

    # MAP initialisation parameters
    parser.add_argument("--lr_map", type=float, default=1e-2, help="MAP learning rate")
    parser.add_argument(
        "--lr_map_shear_factor",
        type=float,
        default=1.0,
        help="Multiplier for g1,g2 learning rate relative to lr_map.",
    )
    parser.add_argument(
        "--map_optimizer",
        type=str,
        default="adam",
        choices=["adam", "adafactor"],
        help="Optimizer for MAP estimation.",
    )
    parser.add_argument(
        "--n_steps_map", type=int, default=5000, help="Number of steps for MAP"
    )
    parser.add_argument(
        "--n_steps_map_freeze_shear",
        type=int,
        default=0,
        help="Initial MAP steps with g1,g2 frozen (optimize only u/flux, 0=disabled).",
    )
    parser.add_argument(
        "--precomputed_map",
        action="store_true",
        default=False,
        help="Load MAP values from output dir instead of re-running MAP.",
    )
    parser.add_argument(
        "--g_chains_init",
        type=float,
        nargs=2,
        default=None,
        metavar=("G1", "G2"),
        help="Initialize all chains at this (g1, g2) in physical units. Converted to MCMC space via g_prior_sigma/g_prior_scale.",
    )
    parser.add_argument(
        "--u_chains_init",
        type=float,
        default=None,
        help="Override the prior draw of u (flow base latent). If None (default), keep the prior sample. If a number, draw u from N(0, sigma^2) per chain/galaxy/element. Use 0.0 for u=0 exactly.",
    )
    parser.add_argument(
        "--point_estimate",
        action="store_true",
        default=False,
        help="Stop after MAP: save g1,g2 estimates and exit (no MCMC).",
    )

    # Sampler parameters
    parser.add_argument(
        "--sampler",
        type=str,
        default="ghmc",
        choices=["ghmc", "mclmc"],
        help="Sampler to use: ghmc or mclmc.",
    )
    parser.add_argument(
        "--n_warmup", type=int, default=5000, help="Number of warmup steps for MEADS."
    )
    parser.add_argument(
        "--num_chains", type=int, default=10, help="Number of chains for HMC."
    )
    parser.add_argument(
        "--step_size", type=float, default=None, help="Step size for HMC."
    )
    parser.add_argument(
        "--mclmc_L",
        type=float,
        default=None,
        help="MCLMC trajectory length L. If both --mclmc_L and --step_size are set, skips adaptation entirely.",
    )
    parser.add_argument(
        "--mclmc_inv_mass_shear",
        type=float,
        default=None,
        help="Diagonal inverse mass matrix value for g1/g2 in MCLMC (all other params use 1.0). "
        "Set to the expected posterior variance of g1/g2 in MCMC sampling space, "
        "e.g. g_scale/sqrt(Ngal). "
        "Default None uses a scalar mass matrix (all params equal).",
    )
    parser.add_argument(
        "--mclmc_inv_mass_file",
        type=str,
        default=None,
        help="Path to .npy file with diagonal inverse mass matrix for MCLMC. "
        "Overrides --mclmc_inv_mass_shear. Use with --mclmc_L and --step_size to skip adaptation.",
    )
    parser.add_argument(
        "--num", type=int, default=20, help="Number of batch iterations."
    )
    parser.add_argument(
        "--num_steps", type=int, default=10000, help="Number of steps for sampling"
    )

    # Output parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../outputs",
        help="Directory to save outputs.",
    )
    parser.add_argument(
        "--save_samples",
        action="store_true",
        default=False,
        help="Save MCMC samples (.npz).",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        default=False,
        help="Save diagnostic plots (.png).",
    )
    parser.add_argument(
        "--save_data",
        action="store_true",
        default=False,
        help="Save intermediate data (radio_data.npy, radio_psf_mask.npy, radio_init_val.npy, radio_map_val.npy).",
    )
    parser.add_argument(
        "--plot_chains",
        type=str,
        default="both",
        help="Plot chains: samples, scaled, both or none. Default: both.",
    )

    return parser


def _validate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Cross-argument validation that argparse cannot express on its own.

    Uses ``parser.error(...)`` rather than ``assert`` so it survives
    ``python -O`` (which strips ``assert`` statements) and exits with the
    conventional CLI status code 2.
    """
    if args.data_profile == "VAE":
        # gen_gal_dataset needs an AE with a trained encoder for whitening.
        # Either --data_vae_path or --vae_path (fallback) must be set.
        if args.data_vae_path is None and args.vae_path is None:
            parser.error(
                "--data_vae_path or --vae_path required when --data_profile=VAE"
            )
        if args.data_vae_epoch is None and args.vae_epoch is None:
            parser.error(
                "--data_vae_epoch or --vae_epoch required when --data_profile=VAE"
            )

    if args.use_flow:
        if args.flow_path is None:
            parser.error("--flow_path required when --use_flow is set")
        if args.flow_epoch is None:
            parser.error("--flow_epoch required when --use_flow is set")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI args from ``argv`` (default: ``sys.argv[1:]``) and validate.

    Cross-argument validation lives in ``_validate`` and runs before the
    namespace is returned, so the rest of the pipeline can assume the args
    are internally consistent.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate(parser, args)
    return args


if __name__ == "__main__":
    # Allow ``python -m shearest.cli --help`` for quick inspection.
    parse_args()
