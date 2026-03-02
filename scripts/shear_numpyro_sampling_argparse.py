import warnings
from functools import partial
from pathlib import Path

import blackjax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax
import equinox as eqx
import blackjax.adaptation.mclmc_adaptation as mclmc_adj
from einops import rearrange
from numpyro.handlers import seed, trace

warnings.filterwarnings("ignore")

import json
import os
import sys

import jax_galsim as galsim # type: ignore

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from argparse import Namespace

import corner

from src.shearest.data_gen_utils import gen_gal_dataset
from src.shearest.func_utils import stack_2_complex, to_unit_disk
from src.shearest.model_utils import model_fn, model_fn_VAE, model_fn_VAE_flow
from src.shearest.psf_utils import compute_radio_uv_mask
from src.shearest.posterior_utils import fit_gmm, save_gmm, plot_gmm_contours

from pshear.utils import load_galaxy_autoencoder # type: ignore
from pshear.nn.flow import make_latent_flow # type: ignore
import yaml


def load_flow_legacy(model_path, epoch):
    """Load flow saved with old flowjax/equinox (sequential numpy format).

    Handles version mismatches between flowjax 13 (checkpoint) and newer
    versions by skipping extra wrapper leaves (shape/cond_shape) that were
    added in later flowjax versions.
    """
    with open(model_path / 'config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Create model skeleton
    flow = make_latent_flow(key=jax.random.key(0), **config)

    # Read checkpoint arrays
    checkpoint_path = model_path / f'model_checkpoint_{epoch}.eqx'
    arrays = []
    with open(checkpoint_path, 'rb') as f:
        while True:
            try:
                arrays.append(np.load(f, allow_pickle=False))
            except Exception:
                break

    # Map arrays to leaves, skipping functions and _dummy
    leaves, treedef = jax.tree_util.tree_flatten(flow)
    flat = jax.tree_util.tree_leaves_with_path(flow)

    serializable_indices = []
    for i, (p, leaf) in enumerate(flat):
        if callable(leaf) and not isinstance(leaf, (jnp.ndarray, np.ndarray)):
            continue
        if '_dummy' in str(p):
            continue
        serializable_indices.append(i)

    # If there are more skeleton leaves than checkpoint arrays, skip extra
    # shape/cond_shape integer leaves added by newer flowjax versions.
    if len(serializable_indices) > len(arrays):
        filtered = []
        for idx in serializable_indices:
            path_str = str(flat[idx][0])
            leaf = flat[idx][1]
            # Skip shape/cond_shape wrapper leaves not present in old checkpoints
            is_shape_leaf = ('shape' in path_str or 'cond_shape' in path_str)
            is_wrapper_int = isinstance(leaf, int) and is_shape_leaf
            if not is_wrapper_int:
                filtered.append(idx)
        serializable_indices = filtered

    assert len(serializable_indices) == len(arrays), \
        f'Leaf count mismatch: {len(serializable_indices)} vs {len(arrays)} arrays'

    new_leaves = list(leaves)
    for arr_idx, leaf_idx in enumerate(serializable_indices):
        old_leaf = leaves[leaf_idx]
        new_val = arrays[arr_idx]
        if isinstance(old_leaf, int):
            new_leaves[leaf_idx] = int(new_val.item())
        elif isinstance(old_leaf, (jnp.ndarray, np.ndarray)):
            new_leaves[leaf_idx] = jnp.array(new_val)
        else:
            new_leaves[leaf_idx] = new_val

    return treedef.unflatten(new_leaves)


# ### Simulation parameters
# Ngal = 100
# Npx = 128
# pixel_scale = 0.15 # in arcsec/pixel
# fov_size = Npx * pixel_scale / 3600 # in degrees
# noise_uv = .004
# params_dir = '../data/trecs_gal_params.npy'
# g1_true = -0.05
# g2_true = 0.05
# ell_sigma = .5
# ell_scale = .3
# g_sigma = 1.0
# g_scale = .3
# sersic_index = 1.

# ### radio PSF parameters
# n_antenna = 50
# E_lim = 50e3
# N_lim = 50e3
# track_time=10
# n_times=4
# f=1.4e9
# df=1e8
# n_freqs=4
# radio_array_seed = 123

# ### Model function params
# ell_prior_sigma = .5
# ell_prior_scale = .3
# g_prior_sigma = 1.0
# g_prior_scale = .3
# hlr_prior_sigma = 2.0
# hlr_prior_offset = 1.
# hlr_prior_scale = 1/1.4
# hlr_prior_min = .2
# flux_prior_sigma = 2.0
# flux_prior_offset = 0.
# flux_prior_scale = 1/15.
# flux_prior_min = .05

# ### Sampler params
# # MAP params
# lr_map = 3e-3
# n_steps_map = 5_000
# # MEADS warmup params
# n_warmup = 500
# # HMC params
# num_chains = 10
# step_size = 0.005
# # batch iterations
# num = 20
# num_steps = 10_000
# save_samples = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Ngal", type=int, default=100, help="Number of galaxies")
    parser.add_argument("--Npx", type=int, default=128, help="Image size in pixels")
    parser.add_argument(
        "--pixel_scale", type=float, default=0.15, help="Pixel scale in arcsec/pixel"
    )
    parser.add_argument("--noise_uv", type=float, default=0.004, help="UV noise level")
    parser.add_argument(
        "--trecs_data_path",
        type=str,
        default=None,
        help="Galaxy, hlr and flux fit over the TRECS catalog (trecs_gal_params.npy)",
    )
    parser.add_argument(
        "--deepshape_data_path",
        type=str,
        default=None,
        help="Path to the DeepShape dataset (val_set_rivi.h5)",
    )
    parser.add_argument(
        "--cosmos_data_path",
        type=str,
        default=None,
        help="Path to the COSMOS dataset 23.5 (for real galaxy images)",
    )
    parser.add_argument(
        "--cosmos_sample",
        type=str,
        default="23.5",
        help="COSMOS dataset sample to use: 23.5 or 25.2",
    )
    parser.add_argument(
        "--data_profile", type=str, default="exp", help="Galaxy dataset profile type: exp, sersic, spergel or real"
    )
    parser.add_argument(
        "--g1_true", type=float, default=-0.05, help="True g1 shear value"
    )
    parser.add_argument(
        "--g2_true", type=float, default=0.05, help="True g2 shear value"
    )
    parser.add_argument(
        "--ell_sigma", type=float, default=1.0, help="Ellipticity prior sigma"
    )
    parser.add_argument(
        "--ell_scale", type=float, default=0.2, help="Ellipticity prior scale"
    )
    parser.add_argument("--g_sigma", type=float, default=1.0, help="Shear prior sigma")
    parser.add_argument("--g_scale", type=float, default=0.1, help="Shear prior scale")
    parser.add_argument("--sersic_index", type=float, default=None, help="Sersic index")
    parser.add_argument("--antenna_type", type=str, default="random", help="Antenna type: random or file")
    parser.add_argument("--antenna_file", type=str, default=None, help="Path to antenna file if antenna_type is file")
    parser.add_argument("--uv_mask_weighting", type=str, default="binary", help="UV weighting: binary or histogram")
    parser.add_argument("--n_antenna", type=int, default=50, help="Number of antennas")
    parser.add_argument("--E_lim", type=float, default=50e3, help="East limit")
    parser.add_argument("--N_lim", type=float, default=50e3, help="North limit")
    parser.add_argument("--track_time", type=float, default=10, help="Track time")
    parser.add_argument("--t0", type=float, default=0, help="Start time")
    parser.add_argument("--n_times", type=int, default=4, help="Number of times")
    parser.add_argument("--f", type=float, default=1.4e9, help="Frequency")
    parser.add_argument("--df", type=float, default=1e8, help="Frequency bandwidth")
    parser.add_argument(
        "--n_freqs", type=int, default=1, help="Number of frequency channels"
    )
    parser.add_argument(
        "--radio_array_seed",
        type=int,
        default=123,
        help="Random seed for the radio array generation",
    )
    parser.add_argument("--model_profile", type=str, default="exp", help="Model profile type: exp, spergel or VAE")
    parser.add_argument(
        "--ell_prior_sigma", type=float, default=1.0, help="Ellipticity prior sigma"
    )
    parser.add_argument(
        "--ell_prior_scale", type=float, default=0.2, help="Ellipticity prior scale"
    )
    parser.add_argument(
        "--g_prior_sigma", type=float, default=1.0, help="Shear prior sigma"
    )
    parser.add_argument(
        "--g_prior_scale", type=float, default=0.1, help="Shear prior scale"
    )
    parser.add_argument(
        "--hlr_prior_sigma",
        type=float,
        default=1.0,
        help="Half-light radius prior sigma",
    )
    parser.add_argument(
        "--hlr_prior_min", type=float, default=0.1, help="Half-light radius prior min"
    )
    parser.add_argument(
        "--hlr_prior_max", type=float, default=3.0, help="Half-light radius prior max"
    )
    parser.add_argument(
        "--flux_prior_sigma", type=float, default=1.0, help="Flux prior sigma"
    )
    parser.add_argument(
        "--flux_prior_min", type=float, default=0.03, help="Flux prior min"
    )
    parser.add_argument(
        "--flux_prior_max", type=float, default=0.25, help="Flux prior max"
    )
    parser.add_argument("--latent_dim", type=int, default=4, help="Latent dimension for VAE, z.shape -> (latent_dim, latent_dim).")
    parser.add_argument("--latent_mean", type=float, default=0., help="Latent representation mean value.")
    parser.add_argument("--vae_path", type=str, default=None, help="Path to the trained VAE model.")
    parser.add_argument("--vae_epoch", type=int, default=None, help="Epoch of the trained VAE model.")
    parser.add_argument("--vae_model_inference_mode", type=str, default="parallel", help="VAE model inference mode: parallel, sequential or batch.")
    parser.add_argument("--vae_inference_batch_size", type=int, default=1, help="VAE inference batch size if using batch mode.")
    parser.add_argument("--use_dropout", action="store_true", help="Enable VAE dropout during inference (disabled by default for deterministic decoding).")
    parser.add_argument("--vae_precision", type=str, default="float16", choices=["float32", "float16"], help="VAE decoder weight precision. float16 gives ~2x speedup on V100 GPU.")
    parser.add_argument("--use_flow", action="store_true", help="Enable normalizing flow reparameterization of VAE latent space.")
    parser.add_argument("--flow_path", type=str, default=None, help="Path to the trained flow model directory.")
    parser.add_argument("--flow_epoch", type=int, default=None, help="Epoch of the trained flow model checkpoint.")
    parser.add_argument("--flow_condition", type=float, nargs=3, default=[21.69, 15.90, 0.60], help="Fixed conditioning values [mag_auto, flux_radius, zphot] for the flow.")
    parser.add_argument("--pixel_scale_vae", type=float, default=0.03, help="Pixel scale for VAE images, default: HST pixel scale (0.03 arcsec/pixel).")
    parser.add_argument("--lr_map", type=float, default=1e-2, help="MAP learning rate")
    parser.add_argument("--lr_map_shear_factor", type=float, default=1.0, help="Multiplier for g1,g2 learning rate relative to lr_map")
    parser.add_argument("--map_optimizer", type=str, default="adam", choices=["adam", "adafactor"], help="Optimizer for MAP estimation")
    parser.add_argument(
        "--n_steps_map", type=int, default=5000, help="Number of steps for MAP"
    )
    parser.add_argument("--n_steps_map_freeze_shear", type=int, default=0, help="Initial MAP steps with g1,g2 frozen (optimize only u/flux, 0=disabled)")
    parser.add_argument("--point_estimate", action="store_true", default=False, help="Stop after MAP: save g1,g2 estimates and exit (no MCMC)")
    parser.add_argument("--sampler", type=str, default="ghmc", help="Sampler to use: ghmc or mclmc")
    parser.add_argument(
        "--n_warmup", type=int, default=5000, help="Number of warmup steps for MEADS"
    )
    parser.add_argument(
        "--num_chains", type=int, default=10, help="Number of chains for HMC"
    )
    parser.add_argument(
        "--step_size", type=float, default=None, help="Step size for HMC"
    )
    parser.add_argument(
        "--mclmc_L", type=float, default=None, help="MCLMC trajectory length L. If both --mclmc_L and --step_size are set, skips adaptation entirely."
    )
    parser.add_argument(
        "--num", type=int, default=20, help="Number of batch iterations"
    )
    parser.add_argument(
        "--num_steps", type=int, default=10000, help="Number of steps for sampling"
    )
    parser.add_argument(
        "--save_samples", action="store_true", default=False, help="Save MCMC samples (.npz)"
    )
    parser.add_argument(
        "--save_plots", action="store_true", default=False, help="Save diagnostic plots (.png)"
    )
    parser.add_argument(
        "--save_data", action="store_true", default=False, help="Save intermediate data (radio_data.npy, radio_psf_mask.npy, radio_init_val.npy, radio_map_val.npy)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: set seed randomly)",
    )
    parser.add_argument(
        "--id", type=str, default=None, help="Unique identifier for the run"
    )
    parser.add_argument(
        "--output_dir", type=str, default="../outputs", help="Directory to save outputs"
    )
    parser.add_argument(
        "--args",
        type=str,
        default=None,
        help="Absolute path to a json file with arguments",
    )
    parser.add_argument(
        "--plot_chains", type=str, default="both", help="Plot chains: samples, scaled, both or none. Default: both."
    )

    args = parser.parse_args()
    # If using arguments from file, load them.
    if args.args is not None:
        id_ = args.id
        out_dir_ = args.output_dir
        with open(args.args, "r") as f:
            args_dict = json.load(f)
        args = Namespace(**args_dict)
        args.id = id_
        args.output_dir = out_dir_

    fov_size = args.Npx * args.pixel_scale / 3600  # in degrees

    # create output folder
    out_dir = args.output_dir
    if args.id is not None:
        out_dir = os.path.join(args.output_dir, args.id)
    os.makedirs(out_dir, exist_ok=True)

    # create log file
    log_file = open(os.path.join(out_dir, "radio_sampling.log"), "w")

    # print parameters to log file
    print(f"Ngal: {args.Ngal}", file=log_file)
    print(f"Npx: {args.Npx}", file=log_file)
    print(f"pixel_scale: {args.pixel_scale}", file=log_file)
    print(f"fov_size: {fov_size}", file=log_file)
    print(f"noise_uv: {args.noise_uv}", file=log_file)
    print(f"g1_true: {args.g1_true}", file=log_file)
    print(f"g2_true: {args.g2_true}", file=log_file)
    print(f"Ellipticity prior scale: {args.ell_scale}", file=log_file)
    print(f"Shear prior scale: {args.g_scale}", file=log_file)

    # Compute the radio PSF
    uv_pos, mask, psf = compute_radio_uv_mask(
        n_antenna=args.n_antenna,
        E_lim=args.E_lim,
        N_lim=args.N_lim,
        Npx=args.Npx,
        fov_size=fov_size,
        track_time=args.track_time,
        t_0=args.t0,
        n_times=args.n_times,
        f=args.f,
        df=args.df,
        n_freqs=args.n_freqs,
        seed=args.radio_array_seed,
        antenna=args.antenna_type,
        antenna_file=args.antenna_file,
        uv_mask_weighting=args.uv_mask_weighting,
    )

    if args.save_plots:
        plt.subplots(1, 3, figsize=(12, 4))
        plt.subplot(131)
        plt.imshow(np.real(mask))
        plt.title("UV mask")
        plt.colorbar()
        plt.subplot(132)
        plt.imshow(psf)
        plt.title("Radio PSF")
        plt.colorbar()
        plt.subplot(133)
        plt.imshow(
            galsim.Gaussian(flux=1.0, sigma=0.2)
            .drawImage(nx=args.Npx, ny=args.Npx, scale=args.pixel_scale)
            .array
        )
        plt.title("Gaussian PSF")
        plt.colorbar()
        plt.savefig(os.path.join(out_dir, "radio_psf.png"))

    # Init seed
    if args.seed is None:
        args.seed = np.random.randint(1, 1e6)
    print(f"Random seed: {args.seed}")
    print(f"Random seed: {args.seed}", file=log_file)
    key = jax.random.PRNGKey(args.seed)

    # Generate observations
    model_data_gen = partial(
        gen_gal_dataset,
        Ngal=args.Ngal,
        Npx=args.Npx,
        pixel_scale=args.pixel_scale,
        uv_pos=uv_pos,
        noise_uv=args.noise_uv,
        TRECS_fit_dir=args.trecs_data_path,
        deepshape_dataset_dir=args.deepshape_data_path,
        cosmos_dataset_dir=args.cosmos_data_path,
        cosmos_sample=args.cosmos_sample,
        ell_scale=args.ell_scale,
        g1=args.g1_true,
        g2=args.g2_true,
        profile_type=args.data_profile,
        n=args.sersic_index,
    )
    seeded_model_data_gen = seed(model_data_gen, key)
    # Conditioning model to generate observation with [g1, g2]
    # conditionned_model = condition(
    #     seeded_model_data_gen,
    #     {
    #         "g1": args.g1_true * jnp.ones((1,)) / (args.g_scale / args.g_sigma),
    #         "g2": args.g2_true * jnp.ones((1,)) / (args.g_scale / args.g_sigma),
    #     },
    # )
    data, data_params = seeded_model_data_gen()

    # Save the data
    if args.save_data:
        np.save(os.path.join(out_dir, "radio_data.npy"), data)
        np.save(os.path.join(out_dir, "radio_data_params.npy"), data_params)
        np.save(os.path.join(out_dir, "radio_psf_mask.npy"), mask)

    key, subkey = jax.random.split(key)

    # Init model for sampling
    if args.model_profile == "VAE":
        # load autoencoder
        VAE_PATH = Path(args.vae_path)
        ae = load_galaxy_autoencoder(VAE_PATH, epoch=args.vae_epoch)
        # Start in float32 for MAP stability; convert to float16 for MCMC later
        jitted_decode = eqx.filter_jit(lambda z, key: ae.decode(z, key=key))
        #
        gsparams = galsim.GSParams(
            minimum_fft_size=128,
            folding_threshold=5e-3,
            maxk_threshold=1e-3,
)
        # Load normalizing flow if requested
        flow_forward = None
        flow_condition = None
        if args.use_flow:
            assert args.flow_path is not None, "--flow_path required when --use_flow is set"
            assert args.flow_epoch is not None, "--flow_epoch required when --use_flow is set"
            flow = load_flow_legacy(Path(args.flow_path), epoch=args.flow_epoch)
            flow_forward = eqx.filter_jit(lambda u, c: flow.flow.bijection.transform(u, c))
            flow_condition = jnp.array(args.flow_condition)
            print(f"Loaded flow from {args.flow_path} epoch {args.flow_epoch}")
            print(f"Flow condition: {flow_condition}")
            print(f"Loaded flow from {args.flow_path} epoch {args.flow_epoch}", file=log_file)
            print(f"Flow condition: {flow_condition}", file=log_file)

        # Initialize the forward model
        if args.use_flow:
            model = partial(
                model_fn_VAE_flow,
                Ngal=args.Ngal,
                Npx=args.Npx,
                pixel_scale_radio=args.pixel_scale,
                pixel_scale_vae=args.pixel_scale_vae,
                uv_pos=uv_pos,
                noise_uv=args.noise_uv,
                obs=data,
                g_sigma=args.g_prior_sigma,
                g_scale=args.g_prior_scale,
                flux_sigma=args.flux_prior_sigma,
                flux_max=args.flux_prior_max,
                flux_min=args.flux_prior_min,
                latent_dim=args.latent_dim,
                jitted_decode=jitted_decode,
                gsparams=gsparams,
                run_type=args.vae_model_inference_mode,
                batch_size=args.vae_inference_batch_size,
                use_dropout=args.use_dropout,
                flow_forward=flow_forward,
                flow_condition=flow_condition,
            )
        else:
            model = partial(
                model_fn_VAE,
                Ngal=args.Ngal,
                Npx=args.Npx,
                pixel_scale_radio=args.pixel_scale,
                pixel_scale_vae=args.pixel_scale_vae,
                uv_pos=uv_pos,
                noise_uv=args.noise_uv,
                obs=data,
                g_sigma=args.g_prior_sigma,
                g_scale=args.g_prior_scale,
                flux_sigma=args.flux_prior_sigma,
                flux_max=args.flux_prior_max,
                flux_min=args.flux_prior_min,
                latent_dim=args.latent_dim,
                latent_mean=args.latent_mean,
                jitted_decode=jitted_decode,
                gsparams=gsparams,
                run_type=args.vae_model_inference_mode,
                batch_size=args.vae_inference_batch_size,
                use_dropout=args.use_dropout,
            )
    else:
        model = partial(
            model_fn,
            Ngal=args.Ngal,
            Npx=args.Npx,
            pixel_scale=args.pixel_scale,
            uv_pos=uv_pos,
            noise_uv=args.noise_uv,
            obs=data,
            ell_sigma=args.ell_prior_sigma,
            ell_scale=args.ell_prior_scale,
            g_sigma=args.g_prior_sigma,
            g_scale=args.g_prior_scale,
            hlr_sigma=args.hlr_prior_sigma,
            hlr_max=args.hlr_prior_max,
            hlr_min=args.hlr_prior_min,
            flux_sigma=args.flux_prior_sigma,
            flux_max=args.flux_prior_max,
            flux_min=args.flux_prior_min,
            profile_type=args.model_profile,
        )
    # seeded_model = seed(model, subkey)

    if args.save_plots:
        # Plot 100 observations
        data_complex = []
        for vis in stack_2_complex(data, batch=True):
            img_aux = np.zeros_like(mask)
            img_aux[uv_pos] = vis
            data_complex.append(img_aux)
        if args.Ngal >= 100:
            data_ = rearrange(data_complex[:100], "(n1 n2) h w -> (n1 h) (n2 w)", n1=10, n2=10)
        else:
            n1 = int(np.ceil(np.sqrt(args.Ngal)))
            n2 = int(np.ceil(np.sqrt(args.Ngal)))
            data_ = rearrange(data_complex[:int(n1*n2)], "(n1 n2) h w -> (n1 h) (n2 w)", n1=n1, n2=n2)
        plt.figure(figsize=(10, 10))
        plt.imshow(np.abs(data_), vmin=np.min(np.abs(data_)), vmax=np.max(np.abs(data_)))
        print("Data shape:", data_.shape)
        print("Data max:", np.max(np.abs(data_)))
        print(f"Data max: {np.max(np.abs(data_))}", file=log_file)
        plt.colorbar()
        plt.savefig(os.path.join(out_dir, "radio_data.png"))

        # Plot a random galaxy
        plt.subplots(1, 2, figsize=(12, 4))
        plt.subplot(121)
        idx = np.random.randint(0, args.Ngal)
        plt.imshow(np.abs(data_complex[idx]))
        plt.title(f"Observed galaxy {idx} uv")
        plt.colorbar()
        plt.subplot(122)
        plt.imshow(np.abs(np.fft.ifftshift(np.fft.ifft2(data_complex[idx]))))
        plt.title(f"Observed galaxy {idx} image")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "radio_data_galaxy.png"))

    # Sample parameters from their prior
    def draw_params(key):
        t = trace(seed(model, key)).get_trace()
        return {key: t[key]["value"] for key in t if not key == "obs"}

    keys = jax.random.split(key, args.num_chains)[: args.num_chains]
    init_val_ = jax.vmap(draw_params)(keys)
    if args.save_data:
        np.save(os.path.join(out_dir, "radio_init_val.npy"), init_val_, allow_pickle=True)

    # Get the log prob of the joint distribution, conditioned on data.
    # Seed the model with a fixed key so the log density is deterministic
    # (required for gradient-based MCMC). This provides the PRNG context
    # needed by numpyro.prng_key() inside model_fn_VAE.
    seeded_model = seed(model, jax.random.PRNGKey(0))

    @jax.jit
    def log_prob_fn(params):
        return numpyro.infer.util.log_density(
            seeded_model,
            (),
            {
                "obs": data,
            },
            params,
        )[0]

    print(f"MAP optimizer: {args.map_optimizer}, lr: {args.lr_map} (shear factor: {args.lr_map_shear_factor}x)", file=log_file)
    print(f"MAP number of steps: {args.n_steps_map}", file=log_file)

    map_init_val = init_val_
    nll = lambda params: -log_prob_fn(params)

    # find the MAP for chain initialization
    def find_map(init_params):
        param_labels = {k: 'shear' if k in ('g1', 'g2') else 'default' for k in init_params}
        opt_fn = optax.adam if args.map_optimizer == "adam" else optax.adafactor

        # Phase 0: optimize only u/flux with g1,g2 frozen
        if args.n_steps_map_freeze_shear > 0:
            optimizer_freeze = optax.multi_transform(
                transforms={
                    'shear': optax.set_to_zero(),
                    'default': opt_fn(args.lr_map),
                },
                param_labels=param_labels,
            )
            opt_state_freeze = optimizer_freeze.init(init_params)

            def update_step_freeze(carry, xs):
                params, opt_state = carry
                loss, grads = jax.value_and_grad(nll)(params)
                updates, opt_state = optimizer_freeze.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                return (params, opt_state), (loss, params["g1"], params["g2"])

            (init_params, _), (losses_f, g1_trace_f, g2_trace_f) = jax.lax.scan(
                update_step_freeze, (init_params, opt_state_freeze), length=args.n_steps_map_freeze_shear
            )

        # Phase 1: joint optimization of all parameters
        optimizer = optax.multi_transform(
            transforms={
                'shear': opt_fn(args.lr_map * args.lr_map_shear_factor),
                'default': opt_fn(args.lr_map),
            },
            param_labels=param_labels,
        )

        opt_state = optimizer.init(init_params)

        def update_step(carry, xs):
            params, opt_state = carry
            loss, grads = jax.value_and_grad(nll)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), (loss, params["g1"], params["g2"])

        (params, _), (losses, g1_trace, g2_trace) = jax.lax.scan(
            update_step, (init_params, opt_state), length=args.n_steps_map
        )

        # Concatenate traces
        if args.n_steps_map_freeze_shear > 0:
            losses = jnp.concatenate([losses_f, losses])
            g1_trace = jnp.concatenate([g1_trace_f, g1_trace])
            g2_trace = jnp.concatenate([g2_trace_f, g2_trace])

        return params, losses, g1_trace, g2_trace

    map_results = jax.vmap(find_map)(map_init_val)
    init_val, map_losses, map_g1_trace, map_g2_trace = map_results

    # Rescale g1,g2 traces to physical units
    g_rescale = args.g_scale / args.g_sigma
    map_g1_trace_phys = map_g1_trace * g_rescale
    map_g2_trace_phys = map_g2_trace * g_rescale

    # Print MAP diagnostics
    print(
        init_val["g1"] * g_rescale,
        init_val["g2"] * g_rescale,
    )
    print(
        f"Initial guess: g1={init_val['g1']*g_rescale}, g2={init_val['g2']*g_rescale}",
        file=log_file,
    )
    print(f"MAP final loss (per chain): {map_losses[:, -1]}", file=log_file)

    if args.save_plots:
        # Plot MAP convergence: loss and g1,g2 evolution
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        total_map_steps = args.n_steps_map_freeze_shear + args.n_steps_map
        steps = jnp.arange(total_map_steps)

        # Loss (log scale, shift to ensure positive values)
        loss_min = jnp.min(map_losses)
        loss_offset = jnp.where(loss_min < 0, jnp.abs(loss_min) + 1.0, 0.0)
        for c in range(map_losses.shape[0]):
            axes[0].plot(steps, map_losses[c] + loss_offset, alpha=0.7, label=f"chain {c}")
        axes[0].set_yscale("log")
        if args.n_steps_map_freeze_shear > 0:
            axes[0].axvline(args.n_steps_map_freeze_shear, color="k", ls=":", alpha=0.5, label="unfreeze g")
        axes[0].set_xlabel("MAP step")
        ylabel = "Loss (NLL)" if loss_offset == 0 else f"Loss (NLL + {loss_offset:.1f})"
        axes[0].set_ylabel(ylabel)
        axes[0].set_title("MAP loss")
        axes[0].legend(fontsize=7)

        # g1
        for c in range(map_g1_trace_phys.shape[0]):
            axes[1].plot(steps, map_g1_trace_phys[c], alpha=0.7, label=f"chain {c}")
        axes[1].axhline(args.g1_true, color="k", ls="--", label="true")
        if args.n_steps_map_freeze_shear > 0:
            axes[1].axvline(args.n_steps_map_freeze_shear, color="k", ls=":", alpha=0.5, label="unfreeze g")
        axes[1].set_xlabel("MAP step")
        axes[1].set_ylabel("g1")
        axes[1].set_title("g1 evolution")
        axes[1].legend(fontsize=7)

        # g2
        for c in range(map_g2_trace_phys.shape[0]):
            axes[2].plot(steps, map_g2_trace_phys[c], alpha=0.7, label=f"chain {c}")
        axes[2].axhline(args.g2_true, color="k", ls="--", label="true")
        if args.n_steps_map_freeze_shear > 0:
            axes[2].axvline(args.n_steps_map_freeze_shear, color="k", ls=":", alpha=0.5, label="unfreeze g")
        axes[2].set_xlabel("MAP step")
        axes[2].set_ylabel("g2")
        axes[2].set_title("g2 evolution")
        axes[2].legend(fontsize=7)

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "map_convergence.png"), dpi=150)
        plt.close(fig)
    if args.save_data:
        np.save(os.path.join(out_dir, "radio_map_val.npy"), init_val, allow_pickle=True)

    if args.point_estimate:
        g1_estimates = init_val["g1"] * g_rescale
        g2_estimates = init_val["g2"] * g_rescale
        np.save(os.path.join(out_dir, "map_shear_estimates.npy"), jnp.stack([g1_estimates, g2_estimates], axis=-1))
        print(f"Point estimate g1 (per chain): {g1_estimates}")
        print(f"Point estimate g2 (per chain): {g2_estimates}")
        print(f"Point estimate g1 mean: {jnp.mean(g1_estimates):.6f}, g2 mean: {jnp.mean(g2_estimates):.6f}")
        print(f"True values: g1={args.g1_true}, g2={args.g2_true}")
        print(f"Point estimate saved to {out_dir}", file=log_file)
        log_file.close()
        sys.exit(0)

    if args.save_plots:
        # Plot the initial guess for the shear
        plt.figure()
        plt.scatter(
            init_val_["g1"] * (args.g_scale / args.g_sigma),
            init_val_["g2"] * (args.g_scale / args.g_sigma),
            label="Initial guess",
        )
        plt.scatter(
            init_val["g1"] * (args.g_scale / args.g_sigma),
            init_val["g2"] * (args.g_scale / args.g_sigma),
            label="MAP estimate",
        )
        plt.scatter(args.g1_true, args.g2_true, color="red", label="True shear")
        plt.xlabel("g1")
        plt.ylabel("g2")
        plt.title("Initial guess for the shear")
        plt.legend()
        # plt.show()
        plt.savefig(os.path.join(out_dir, "radio_initial_guess.png"))

    # Use the the MEADS algorithm for parallel chains on GPUs
    """
    - https://proceedings.mlr.press/v151/hoffman22a/hoffman22a.pdf
    - https://blackjax-devs.github.io/blackjax/autoapi/blackjax/adaptation/meads_adaptation/index.html
    - https://blackjax-devs.github.io/blackjax/autoapi/blackjax/mcmc/ghmc/index.html
    """

    key_warmup, key_sample = jax.random.split(key)

    if args.sampler == "ghmc":
        print("Using GHMC sampler with MEADS adaptation", file=log_file)
        warmup = blackjax.meads_adaptation(
            log_prob_fn,
            num_chains=args.num_chains,
        )

        (last_states, parameters), _ = warmup.run(
            key_warmup,
            init_val,
            num_steps=args.n_warmup,
        )

        print("Step size:", parameters["step_size"])
        print(f"Step size: {parameters['step_size']}", file=log_file)
        if args.step_size is not None:
            parameters["step_size"] = args.step_size
            print("Set step size to:", parameters["step_size"])
            print(f"Set step size to: {parameters['step_size']}", file=log_file)
        print(parameters.keys(), file=log_file)
        print(parameters, file=log_file)
        if args.save_data:
            np.save(
                os.path.join(out_dir, "radio_meads_warmup.npy"),
                last_states.position,
                allow_pickle=True,
            )

        # Convert VAE to float16 for sampling (after adaptation in float32)
        if args.model_profile == "VAE" and args.vae_precision == "float16":
            ae = jax.tree.map(
                lambda x: x.astype(jnp.float16) if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating) else x,
                ae,
            )
            jitted_decode = eqx.filter_jit(lambda z, key: ae.decode(z.astype(jnp.float16), key=key))
            if args.use_flow:
                model = partial(
                    model_fn_VAE_flow,
                    Ngal=args.Ngal, Npx=args.Npx,
                    pixel_scale_radio=args.pixel_scale, pixel_scale_vae=args.pixel_scale_vae,
                    uv_pos=uv_pos, noise_uv=args.noise_uv, obs=data,
                    g_sigma=args.g_prior_sigma, g_scale=args.g_prior_scale,
                    flux_sigma=args.flux_prior_sigma, flux_max=args.flux_prior_max, flux_min=args.flux_prior_min,
                    latent_dim=args.latent_dim,
                    jitted_decode=jitted_decode, gsparams=gsparams,
                    run_type=args.vae_model_inference_mode, batch_size=args.vae_inference_batch_size,
                    use_dropout=args.use_dropout,
                    flow_forward=flow_forward, flow_condition=flow_condition,
                )
            else:
                model = partial(
                    model_fn_VAE,
                    Ngal=args.Ngal, Npx=args.Npx,
                    pixel_scale_radio=args.pixel_scale, pixel_scale_vae=args.pixel_scale_vae,
                    uv_pos=uv_pos, noise_uv=args.noise_uv, obs=data,
                    g_sigma=args.g_prior_sigma, g_scale=args.g_prior_scale,
                    flux_sigma=args.flux_prior_sigma, flux_max=args.flux_prior_max, flux_min=args.flux_prior_min,
                    latent_dim=args.latent_dim, latent_mean=args.latent_mean,
                    jitted_decode=jitted_decode, gsparams=gsparams,
                    run_type=args.vae_model_inference_mode, batch_size=args.vae_inference_batch_size,
                    use_dropout=args.use_dropout,
                )
            seeded_model = seed(model, jax.random.PRNGKey(0))

            @jax.jit
            def log_prob_fn(params):
                return numpyro.infer.util.log_density(
                    seeded_model, (), {"obs": data}, params,
                )[0]

            print("VAE decoder converted to float16 for sampling")

        kernel = blackjax.ghmc(log_prob_fn, **parameters)

    elif args.sampler == "mclmc":
        key_init, key_tune = jax.random.split(key_warmup)
        key_init_chains = jax.random.split(key_init, args.num_chains)

        # Compute dimensionality for initial L and step_size heuristics
        first_chain_init = jax.tree.map(lambda x: x[0], init_val)
        ndim = sum(v.size for v in jax.tree.leaves(first_chain_init))
        initial_L = jnp.sqrt(float(ndim))
        initial_step_size = initial_L / ndim
        print(f"MCLMC init: ndim={ndim}, initial_L={initial_L:.2f}, initial_step_size={initial_step_size:.4f}")

        def mclmc_factory(inverse_mass_matrix):
            return blackjax.mcmc.mclmc.build_kernel(
                logdensity_fn=log_prob_fn,
                inverse_mass_matrix=inverse_mass_matrix,
                integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
            )

        inverse_mass_matrix = 1.0

        temp_kernel = blackjax.mclmc(
            log_prob_fn,
            step_size=initial_step_size,
            L=initial_L,
            inverse_mass_matrix=inverse_mass_matrix,
        )

        # Skip adaptation if both L and step_size are provided
        first_chain_state = temp_kernel.init(first_chain_init, key_init_chains[0])

        if args.mclmc_L is not None and args.step_size is not None:
            print(f"Skipping MCLMC adaptation: using L={args.mclmc_L}, step_size={args.step_size}")
            print(f"Skipping MCLMC adaptation: using L={args.mclmc_L}, step_size={args.step_size}", file=log_file)
            parameters = mclmc_adj.MCLMCAdaptationState(
                L=jnp.array(args.mclmc_L),
                step_size=jnp.array(args.step_size),
                inverse_mass_matrix=jnp.array(inverse_mass_matrix),
            )
        else:
            max_adapt_attempts = 10
            for adapt_attempt in range(1, max_adapt_attempts + 1):
                print(f"MCLMC adaptation attempt {adapt_attempt}/{max_adapt_attempts}...")
                key_tune, key_retry = jax.random.split(key_tune)

                adapted_state, parameters, _ = mclmc_adj.mclmc_find_L_and_step_size(
                                                mclmc_kernel=mclmc_factory,
                                                num_steps=args.n_warmup,
                                                state=first_chain_state,
                                                rng_key=key_retry,
                )

                if parameters.step_size > 0 and parameters.L > 0:
                    break
                print(f"Adaptation failed (step_size={parameters.step_size}, L={parameters.L}), retrying...")

            if parameters.step_size <= 0 or parameters.L <= 0:
                msg = (f"MCLMC adaptation failed after {max_adapt_attempts} attempts: "
                       f"step_size={parameters.step_size}, L={parameters.L}")
                print(msg)
                print(msg, file=log_file)
                log_file.close()
                raise RuntimeError(msg)

            print("Step size:", parameters.step_size)
            print(f"Step size: {parameters.step_size}", file=log_file)
            print("L:", parameters.L)
            print(f"L: {parameters.L}", file=log_file)

        # Convert VAE to float16 for sampling (after adaptation in float32)
        if args.model_profile == "VAE" and args.vae_precision == "float16":
            ae = jax.tree.map(
                lambda x: x.astype(jnp.float16) if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating) else x,
                ae,
            )
            jitted_decode = eqx.filter_jit(lambda z, key: ae.decode(z.astype(jnp.float16), key=key))
            if args.use_flow:
                model = partial(
                    model_fn_VAE_flow,
                    Ngal=args.Ngal, Npx=args.Npx,
                    pixel_scale_radio=args.pixel_scale, pixel_scale_vae=args.pixel_scale_vae,
                    uv_pos=uv_pos, noise_uv=args.noise_uv, obs=data,
                    g_sigma=args.g_prior_sigma, g_scale=args.g_prior_scale,
                    flux_sigma=args.flux_prior_sigma, flux_max=args.flux_prior_max, flux_min=args.flux_prior_min,
                    latent_dim=args.latent_dim,
                    jitted_decode=jitted_decode, gsparams=gsparams,
                    run_type=args.vae_model_inference_mode, batch_size=args.vae_inference_batch_size,
                    use_dropout=args.use_dropout,
                    flow_forward=flow_forward, flow_condition=flow_condition,
                )
            else:
                model = partial(
                    model_fn_VAE,
                    Ngal=args.Ngal, Npx=args.Npx,
                    pixel_scale_radio=args.pixel_scale, pixel_scale_vae=args.pixel_scale_vae,
                    uv_pos=uv_pos, noise_uv=args.noise_uv, obs=data,
                    g_sigma=args.g_prior_sigma, g_scale=args.g_prior_scale,
                    flux_sigma=args.flux_prior_sigma, flux_max=args.flux_prior_max, flux_min=args.flux_prior_min,
                    latent_dim=args.latent_dim, latent_mean=args.latent_mean,
                    jitted_decode=jitted_decode, gsparams=gsparams,
                    run_type=args.vae_model_inference_mode, batch_size=args.vae_inference_batch_size,
                    use_dropout=args.use_dropout,
                )
            seeded_model = seed(model, jax.random.PRNGKey(0))

            @jax.jit
            def log_prob_fn(params):
                return numpyro.infer.util.log_density(
                    seeded_model, (), {"obs": data}, params,
                )[0]

            print("VAE decoder converted to float16 for sampling")

        # Build the final kernel with tuned parameters
        kernel = blackjax.mclmc(log_prob_fn, **parameters._asdict())

        # Initialize all chains with tuned kernel
        last_states = jax.vmap(kernel.init)(init_val, key_init_chains)

        # kernel = blackjax.mclmc(log_prob_fn, step_size=1e-3)
        # state = kernel.init(init_val)

        # da_init, da_update, da_final = blackjax.dual_averaging(
        #     initial_step_size=1e-3,
        #     target_acceptance_rate=None,
        # )

        # da_state = da_init()

        # for _ in range(args.n_warmup):
        #     key_warmup, subkey = jax.random.split(key_warmup)
        #     new_state, info = kernel.step(subkey, state)
        #     log_energy_change = info.log_kinetic_energy_change
        #     da_state = da_update(da_state, log_energy_change)
        #     state = new_state
        
        # step_size = da_final(da_state)
        # print("Step size:", step_size)
        # print(f"Step size: {step_size}", file=log_file)

        # kernel = blackjax.mclmc(log_prob_fn, step_size=step_size)

    else:
        raise ValueError("Sampler not recognized. Use ghmc or mclmc.")

    @partial(jax.jit, static_argnames=("num_steps",))
    def run_hmc(init_states, key, num_steps=1):

        def make_step(state, key):
            state, info = kernel.step(key, state)
            return state, (state, info)

        keys = jax.random.split(key, num_steps)
        last_states, (samples, info) = jax.lax.scan(make_step, init_states, keys)

        return last_states, (samples, info)

    # loop over lax.scan to save GPU memorry
    print(f"Number of chains: {args.num_chains}", file=log_file)
    print(f"Number of loops: {args.num}", file=log_file)
    print(f"Number of steps: {args.num_steps}", file=log_file)
    print(f"Number of samples per chain: {args.num_steps*args.num*2}", file=log_file)

    key_chains = jax.random.split(key_sample, args.num_chains)

    last_states, _ = jax.vmap(lambda init_states, keys: run_hmc(init_states, keys, 1))(
        last_states, key_chains
    )

    sample_list = []

    keys = jax.vmap(jax.random.split, in_axes=(0, None))(key_chains, 2 * args.num)

    for i in range(args.num):
        print("Chain", i + 1, "of", 2 * args.num, "running...")
        last_states, (samples, info) = jax.vmap(
            lambda init_states, keys: run_hmc(init_states, keys, args.num_steps)
        )(last_states, keys[:, i, :])
        sample_list.append(samples)

        # Quick diagnostics: sampler health and shear chain statistics (zero extra compute).
        # MCLMC info has no acceptance_rate; energy_change measures integrator accuracy:
        #   |energy_change| << 1  →  step_size fine
        #   |energy_change| >> 1  →  step_size too large, expect poor mixing
        # GHMC info has acceptance_rate directly (target > 0.6).
        if args.sampler == "mclmc":
            sampler_diag = f"mean|energy_change|={float(jnp.abs(info.energy_change).mean()):.3f}"
        else:
            sampler_diag = f"accept={float(info.acceptance_rate.mean()):.3f}"
        g1_mean = float(samples.position["g1"].mean()) * g_rescale
        g1_std  = float(samples.position["g1"].std())  * g_rescale
        g2_mean = float(samples.position["g2"].mean()) * g_rescale
        g2_std  = float(samples.position["g2"].std())  * g_rescale
        diag = (f"  {sampler_diag} | "
                f"g1={g1_mean:.4f}±{g1_std:.4f} | "
                f"g2={g2_mean:.4f}±{g2_std:.4f}")
        print(diag)
        print(diag, file=log_file)

    samples_ = {
        key: np.concatenate([sample_list[k].position[key] for k in range(args.num)], 1)
        for key in last_states.position
    }
    print("ESS g1", blackjax.diagnostics.effective_sample_size(samples_["g1"][..., 0]))
    print("ESS g2", blackjax.diagnostics.effective_sample_size(samples_["g2"][..., 0]))
    print(
        "ESS flux", blackjax.diagnostics.effective_sample_size(samples_["flux"][..., 0])
    )
    if args.model_profile != "VAE":
        print(
            "ESS hlr", blackjax.diagnostics.effective_sample_size(samples_["hlr"][..., 0])
        )
        print("ESS e1", blackjax.diagnostics.effective_sample_size(samples_["e1"][..., 0]))
        print("ESS e2", blackjax.diagnostics.effective_sample_size(samples_["e2"][..., 0]))
    print("ESS at the end of first loop", file=log_file)
    print(
        "ESS g1",
        blackjax.diagnostics.effective_sample_size(samples_["g1"][..., 0]),
        file=log_file,
    )
    print(
        "ESS g2",
        blackjax.diagnostics.effective_sample_size(samples_["g2"][..., 0]),
        file=log_file,
    )
    print(
        "ESS flux",
        blackjax.diagnostics.effective_sample_size(samples_["flux"][..., 0]),
        file=log_file,
    )
    if args.model_profile != "VAE":
        print(
            "ESS hlr",
            blackjax.diagnostics.effective_sample_size(samples_["hlr"][..., 0]),
            file=log_file,
        )
        
        print(
            "ESS e1",
            blackjax.diagnostics.effective_sample_size(samples_["e1"][..., 0]),
            file=log_file,
        )
        print(
            "ESS e2",
            blackjax.diagnostics.effective_sample_size(samples_["e2"][..., 0]),
            file=log_file,
        )

    # extra chains
    for i in range(args.num):
        print("Extra chain", args.num + i + 1, "of", 2 * args.num, "running...")
        last_states, (samples, info) = jax.vmap(
            lambda init_states, keys: run_hmc(init_states, keys, args.num_steps)
        )(last_states, keys[:, args.num + i, :])
        sample_list.append(samples)

    # concatenates chains
    samples_ = {
        key: np.concatenate(
            [sample_list[k].position[key] for k in range(args.num * 2)], 1
        )
        for key in last_states.position
    }

    # labels = ["hlr", "flux", "r_ell", "angle_ell", "g1", "g2"]
    if args.model_profile == "VAE":
        labels = ["flux", "g1", "g2"]
    else:
        labels = ["hlr", "flux", "e1", "e2", "g1", "g2"]

    if args.save_plots:
        fig, axes = plt.subplots(len(labels), figsize=(10, 7), sharex=True)
        for i, label in enumerate(labels):
            print(i, label)
            ax = axes[i]
            for k in range(args.num_chains):
                # if label in ["g1", "g2"]:
                #     ax.plot(samples_[label][k,:,0]*0.1, "k", alpha=0.3)
                # else:
                #     ax.plot(samples_[label][k,:,0], "k", alpha=0.3)
                ax.plot(samples_[label][k, :, 0], "k", alpha=0.3)
            ax.set_xlim(0, args.num_steps * args.num * 2)
            ax.set_ylabel(label)
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number")
        if args.plot_chains in ["samples", "both"]:
            plt.savefig(os.path.join(out_dir, "radio_chains.png"))
        plt.close()

        fig, axes = plt.subplots(len(labels), figsize=(10, 7), sharex=True)
        for i, label in enumerate(labels):
            ax = axes[i]
            for k in range(args.num_chains):
                if label == "hlr":
                    # hlr -> jax.nn.softplus(hlr + hlr_offset) * hlr_scale + hlr_min
                    ax.plot(
                        jax.nn.sigmoid(samples_["hlr"][k, :, 0] / args.hlr_prior_sigma)
                        * (args.hlr_prior_max - args.hlr_prior_min)
                        + args.hlr_prior_min,
                        "k",
                        alpha=0.3,
                    )
                if label == "flux":
                    # flux -> jax.nn.softplus(flux + flux_offset) * flux_scale + flux_min
                    ax.plot(
                        jax.nn.sigmoid(samples_["flux"][k, :, 0] / args.flux_prior_sigma)
                        * (args.flux_prior_max - args.flux_prior_min)
                        + args.flux_prior_min,
                        "k",
                        alpha=0.3,
                    )
                if label in ["e1", "e2"]:
                    #  e1, e2 -> to_unit_disk
                    e = jnp.stack(
                        [
                            samples_["e1"][k, :, 0] / args.ell_sigma * args.ell_scale,
                            samples_["e2"][k, :, 0] / args.ell_sigma * args.ell_scale,
                        ],
                        0,
                    )
                    e = to_unit_disk(e)
                    if label == "e1":
                        ax.plot(e[0], "k", alpha=0.3)
                    else:
                        ax.plot(e[1], "k", alpha=0.3)
                if label in ["g1", "g2"]:
                    # g1, g2 -> to_unit_disk
                    g = jnp.stack(
                        [
                            samples_["g1"][k, :, 0] / args.g_sigma * args.g_scale,
                            samples_["g2"][k, :, 0] / args.g_sigma * args.g_scale,
                        ],
                        0,
                    )
                    g = to_unit_disk(g)
                    if label == "g1":
                        ax.plot(g[0], "k", alpha=0.3)
                    else:
                        ax.plot(g[1], "k", alpha=0.3)
                else:
                    pass

            ax.set_xlim(0, args.num_steps * args.num * 2)
            ax.set_ylabel(label)
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number")
        if args.plot_chains in ["scaled", "both"]:
            plt.savefig(os.path.join(out_dir, "radio_chains_scaled.png"))
        plt.close()

        two_truths = np.array([args.g1_true, args.g2_true])
        samples_g = np.concatenate([samples_["g1"], samples_["g2"]], -1).reshape(
            (-1, 2)
        ) * (args.g_scale / args.g_sigma)

        two_cols = ["g_1", "g_2"]
        two_labels = [r"$\gamma_1$", r"$\gamma_2$"]

        fig = plt.figure(figsize=(7, 7))
        fig = corner.corner(samples_g, truths=two_truths, labels=two_labels, fig=fig)
        fig.savefig(os.path.join(out_dir, "radio_corner_g.png"))
        plt.close()

    print("ESS g1", blackjax.diagnostics.effective_sample_size(samples_["g1"][..., 0]))
    print("ESS g2", blackjax.diagnostics.effective_sample_size(samples_["g2"][..., 0]))
    print(
        "ESS flux", blackjax.diagnostics.effective_sample_size(samples_["flux"][..., 0])
    )
    if args.model_profile != "VAE":
        print(
            "ESS hlr", blackjax.diagnostics.effective_sample_size(samples_["hlr"][..., 0])
        )
        
        print("ESS e1", blackjax.diagnostics.effective_sample_size(samples_["e1"][..., 0]))
        print("ESS e2", blackjax.diagnostics.effective_sample_size(samples_["e2"][..., 0]))
    print("ESS at the end of second loop", file=log_file)
    print(
        "ESS g1",
        blackjax.diagnostics.effective_sample_size(samples_["g1"][..., 0]),
        file=log_file,
    )
    print(
        "ESS g2",
        blackjax.diagnostics.effective_sample_size(samples_["g2"][..., 0]),
        file=log_file,
    )
    print(
        "ESS flux",
        blackjax.diagnostics.effective_sample_size(samples_["flux"][..., 0]),
        file=log_file,
    )
    if args.model_profile != "VAE":
        print(
            "ESS hlr",
            blackjax.diagnostics.effective_sample_size(samples_["hlr"][..., 0]),
            file=log_file,
        )
        print(
            "ESS e1",
            blackjax.diagnostics.effective_sample_size(samples_["e1"][..., 0]),
            file=log_file,
        )
        print(
            "ESS e2",
            blackjax.diagnostics.effective_sample_size(samples_["e2"][..., 0]),
            file=log_file,
        )

    flatchain = np.std(samples_["g1"], axis=1) < 1e-4
    print("Flatchains:")
    print(flatchain)
    print("Flatchains:", file=log_file)
    print(flatchain, file=log_file)

    # Compute posterior density estimation shear samples
    g1_scaled = samples_["g1"][np.where(flatchain == False),:]
    g2_scaled = samples_["g2"][np.where(flatchain == False),:]
    samples_g_scaled = np.concatenate([g1_scaled, g2_scaled], -1).reshape((-1,2)) / args.g_sigma * args.g_scale
    # g_mean = np.mean(samples_g_scaled, axis=0)
    # g_std = np.sqrt(np.diag(np.cov(samples_g_scaled, rowvar=False)))
    # print(f"Shear mean: g1={g_mean[0]}, g2={g_mean[1]}")
    # print(f"Shear std: g1={g_std[0]}, g2={g_std[1]}")
    # print(f"Shear mean: g1={g_mean[0]}, g2={g_mean[1]}", file=log_file)
    # print(f"Shear std: g1={g_std[0]}, g2={g_std[1]}", file=log_file)
    # np.savez(
    #     os.path.join(out_dir, "radio_shear_stats.npz"),
    #     g_mean=g_mean,
    #     g_std=g_std,
    #     flatchain=flatchain,
    # )

    # Fit and save GMM posterior density
    print("Fitting GMM to shear posterior...")
    gmm_params = fit_gmm(samples_g_scaled, n_components=5)
    save_gmm(gmm_params, os.path.join(out_dir, "radio_shear_gmm.npz"))
    print(f"GMM saved with {len(gmm_params['weights'])} components")
    print(f"GMM saved with {len(gmm_params['weights'])} components", file=log_file)

    if args.save_plots:
        # Plot GMM contours
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        plot_gmm_contours(
            gmm_params, ax=ax,
            true_g=(args.g1_true, args.g2_true),
        )
        ax.set_title("GMM Posterior Density")
        fig.savefig(os.path.join(out_dir, "gmm_contours.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Save samples
    if args.save_samples:
        print("Saving samples...")
        np.savez(os.path.join(out_dir, "radio_samples.npz"), **samples_)

    # Print arguments
    print("Arguments:", file=log_file)
    for key, value in vars(args).items():
        print(f"  {key}: {value}", file=log_file)

    # Save the arguments
    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # Save log file
    log_file.close()

    print("Done.")


if __name__ == "__main__":
    main()
