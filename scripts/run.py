import warnings
from datetime import datetime
from functools import partial
from pathlib import Path

import blackjax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import optax
import equinox as eqx
import blackjax.adaptation.mclmc_adaptation as mclmc_adj
from einops import rearrange
from jax.flatten_util import ravel_pytree
from numpyro.handlers import seed, trace

warnings.filterwarnings("ignore")

import json
import os
import sys

import jax_galsim as galsim  # type: ignore

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import corner

from src.shearest.cli import parse_args
from src.shearest.logging_setup import setup_logger
from src.shearest.data_gen_utils import gen_gal_dataset
from src.shearest.func_utils import stack_2_complex, to_unit_disk
from src.shearest.model_utils import (
    model_fn,
    model_fn_VAE,
    model_fn_VAE_flow,
    model_fn_composite,
)
from src.shearest.psf_utils import compute_radio_uv_mask
from src.shearest.posterior_utils import fit_gmm, save_gmm, plot_gmm_contours

from pshear.utils import load_galaxy_autoencoder  # type: ignore
from pshear.utils import load_flow  # type: ignore
import yaml


def main():
    args = parse_args()

    fov_size = args.Npx * args.pixel_scale / 3600  # in degrees

    # create output folder
    out_dir = args.output_dir
    if args.id is not None:
        out_dir = os.path.join(args.output_dir, args.id)
    os.makedirs(out_dir, exist_ok=True)

    # Configure the package logger (stdout + radio_sampling.log file handler).
    logger = setup_logger(out_dir)
    t_start = datetime.now()
    logger.info(f"Start: {t_start.strftime('%Y-%m-%d %H:%M:%S')}")

    # Save the arguments
    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # print parameters to log file
    logger.info(f"Ngal: {args.Ngal}")
    logger.info(f"Npx: {args.Npx}")
    logger.info(f"pixel_scale: {args.pixel_scale}")
    logger.info(f"fov_size: {fov_size}")
    logger.info(f"noise_uv (model): {args.noise_uv}")
    logger.info(f"noise_data: {args.noise_data if args.noise_data is not None else f'{args.noise_uv} (fallback to noise_uv)'}")
    logger.info(f"g1_true: {args.g1_true}")
    logger.info(f"g2_true: {args.g2_true}")
    logger.info(f"Ellipticity scale (data gen): {args.ell_scale}")
    logger.info(f"Shear prior scale: {args.g_prior_scale}")

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
        lat=args.array_lat * np.pi / 180,
        dec=args.array_dec * np.pi / 180,
        n_freqs=args.n_freqs,
        seed=args.radio_array_seed,
        antenna=args.antenna_type,
        antenna_file=args.antenna_file,
        uv_mask_weighting=args.uv_mask_weighting,
    )

    if args.save_plots:
        # Plot radio UV mask and PSF
        plt.subplots(1, 2, figsize=(8, 4))
        plt.subplot(121)
        plt.imshow(np.real(mask))
        plt.title("UV mask")
        plt.colorbar()
        plt.subplot(122)
        plt.imshow(psf)
        plt.title("Radio PSF")
        plt.colorbar()
        plt.savefig(os.path.join(out_dir, "radio_psf.png"))

    # Init seed
    if args.seed is None:
        args.seed = np.random.randint(1, 1e6)
    logger.info(f"Random seed: {args.seed}")
    key = jax.random.PRNGKey(args.seed)

    # Generate or load observations
    data_file = os.path.join(out_dir, "radio_data.npy")
    if args.precomputed_map and os.path.exists(data_file):
        logger.info(f"Loading precomputed data from {data_file}")
        data = jnp.array(np.load(data_file))
        data_params = np.load(
            os.path.join(out_dir, "radio_data_params.npy"), allow_pickle=True
        )[()]
    else:
        noise_uv_data = (
            args.noise_data if args.noise_data is not None else args.noise_uv
        )

        data_ae = None
        if args.data_profile == "VAE":
            # Load the data-generation AE (with trained encoder) when
            # --data_profile=VAE. Falls back to the model AE path/epoch if the
            # data-specific args are not set. The encoder is needed, so make sure
            # the path points to a non-distilled checkpoint.
            # Presence of (data_vae_path or vae_path) is guaranteed by cli._validate.
            data_vae_path = args.data_vae_path if args.data_vae_path else args.vae_path
            data_vae_epoch = (
                args.data_vae_epoch
                if args.data_vae_epoch is not None
                else args.vae_epoch
            )
            logger.info(f"Loading data-generation AE from {data_vae_path} epoch {data_vae_epoch}")
            data_ae = load_galaxy_autoencoder(Path(data_vae_path), epoch=data_vae_epoch)
            data_ae = eqx.nn.inference_mode(data_ae, True)

        # Initialise the data generator and generate the observations.
        model_data_gen = partial(
            gen_gal_dataset,
            Ngal=args.Ngal,
            Npx=args.Npx,
            pixel_scale=args.pixel_scale,
            pixel_scale_vae=args.pixel_scale_vae,
            uv_pos=uv_pos,
            noise_uv=noise_uv_data,
            TRECS_fit_dir=args.trecs_data_path,
            deepshape_dataset_dir=args.deepshape_data_path,
            cosmos_dataset_dir=args.cosmos_data_path,
            cosmos_sample=args.cosmos_sample,
            ell_scale=args.ell_scale,
            g1=args.g1_true,
            g2=args.g2_true,
            profile_type=args.data_profile,
            n=args.sersic_index,
            cosmos_seed=args.seed,
            mag_cut=args.mag_cut,
            ae=data_ae,
        )
        seeded_model_data_gen = seed(model_data_gen, key)
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
            # Presence of flow_path and flow_epoch is guaranteed by cli._validate.
            flow = load_flow(Path(args.flow_path), epoch=args.flow_epoch)
            # The flow's base_dist may have learned loc/scale parameters during
            # training, so we must apply base_dist.bijection (Affine) before the
            # main bijection to match what flow.sample() does internally.
            base_bij = flow.flow.base_dist.bijection
            if flow.cond_dim > 0:
                flow_forward = eqx.filter_jit(
                    lambda u, c: flow.flow.bijection.transform(base_bij.transform(u), c)
                )
                flow_condition = jnp.array(args.flow_condition)
            else:
                flow_forward = eqx.filter_jit(
                    lambda u: flow.flow.bijection.transform(base_bij.transform(u))
                )
                flow_condition = None
            logger.info(f"Loaded flow from {args.flow_path} epoch {args.flow_epoch}")
            logger.info(f"Flow condition: {flow_condition}")
            logger.info(f"Loaded flow from {args.flow_path} epoch {args.flow_epoch}")
            logger.info(f"Flow condition: {flow_condition}")

        # Initialize the forward model
        if args.use_flow:
            model = partial(
                model_fn_VAE_flow,
                Ngal=args.Ngal,
                Npx=args.Npx,
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
                latent_sigma=args.latent_sigma,
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
                latent_sigma=args.latent_sigma,
                jitted_decode=jitted_decode,
                gsparams=gsparams,
                run_type=args.vae_model_inference_mode,
                batch_size=args.vae_inference_batch_size,
                use_dropout=args.use_dropout,
            )
    elif args.model_profile == "composite":
        model = partial(
            model_fn_composite,
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
            flux_ratio_max=args.composite_flux_ratio_max,
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
            data_ = rearrange(
                data_complex[:100], "(n1 n2) h w -> (n1 h) (n2 w)", n1=10, n2=10
            )
        else:
            n1 = int(np.ceil(np.sqrt(args.Ngal)))
            n2 = int(np.ceil(np.sqrt(args.Ngal)))
            data_ = rearrange(
                data_complex[: int(n1 * n2)],
                "(n1 n2) h w -> (n1 h) (n2 w)",
                n1=n1,
                n2=n2,
            )
        plt.figure(figsize=(10, 10))
        plt.imshow(
            np.abs(data_), vmin=np.min(np.abs(data_)), vmax=np.max(np.abs(data_))
        )
        logger.info(f'Data shape: {data_.shape}')
        logger.info(f'Data max: {np.max(np.abs(data_))}')
        logger.info(f"Data max: {np.max(np.abs(data_))}")
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

    if args.g_chains_init is not None and "g1" in init_val_:
        g1_init, g2_init = args.g_chains_init
        # Convert from physical to MCMC space (inverse of g_rescale = g_prior_scale/g_prior_sigma)
        g1_mcmc = g1_init * args.g_prior_sigma / args.g_prior_scale
        g2_mcmc = g2_init * args.g_prior_sigma / args.g_prior_scale
        init_val_ = {
            **init_val_,
            "g1": jnp.full_like(init_val_["g1"], g1_mcmc),
            "g2": jnp.full_like(init_val_["g2"], g2_mcmc),
        }
        logger.info(f"g_chains_init: g1={g1_init}, g2={g2_init} (physical) -> g1={g1_mcmc:.4f}, g2={g2_mcmc:.4f} (MCMC space)")

    # Override the prior draw of u (flow base latent) with N(0, sigma^2). When
    # sigma=0, u collapses to zeros. Useful for testing how much of the MAP
    # outcome is driven by the initial u vs by the data.
    if args.u_chains_init is not None and "u" in init_val_:
        key, ukey = jax.random.split(key)
        u_shape = init_val_["u"].shape
        init_val_ = {
            **init_val_,
            "u": args.u_chains_init
            * jax.random.normal(ukey, u_shape, dtype=init_val_["u"].dtype),
        }
        logger.info(f"u_chains_init: u ~ N(0, {args.u_chains_init}^2) (shape {u_shape})")

    if args.save_data:
        np.save(
            os.path.join(out_dir, "radio_init_val.npy"), init_val_, allow_pickle=True
        )

    # Get the log prob of the joint distribution, conditioned on data.
    # Seed the model with a fixed key so the log density is deterministic
    # (required for gradient-based MCMC). This provides the PRNG context
    # needed by numpyro.prng_key() inside model_fn_VAE.
    seeded_model = seed(model, jax.random.PRNGKey(0))

    @jax.jit
    def log_prob_fn(params):
        @jax.checkpoint
        def _log_density(params):
            return numpyro.infer.util.log_density(
                seeded_model,
                (),
                {
                    "obs": data,
                },
                params,
            )[0]

        return _log_density(params)

    logger.info(f"MAP optimizer: {args.map_optimizer}, lr: {args.lr_map} (shear factor: {args.lr_map_shear_factor}x)")
    logger.info(f"MAP number of steps: {args.n_steps_map}")

    map_init_val = init_val_
    nll = lambda params: -log_prob_fn(params)

    # find the MAP for chain initialization
    has_shear = "g1" in map_init_val
    g_rescale = args.g_prior_scale / args.g_prior_sigma

    # Try to load precomputed MAP values
    map_file = os.path.join(out_dir, "radio_map_val.npy")
    if args.precomputed_map and os.path.exists(map_file):
        logger.info(f"Loading precomputed MAP from {map_file}")
        init_val = np.load(map_file, allow_pickle=True)[()]
        # Convert to jax arrays
        init_val = {k: jnp.array(v) for k, v in init_val.items()}
        if has_shear:
            logger.info(f"Loaded MAP: g1={init_val['g1']*g_rescale}, g2={init_val['g2']*g_rescale}")
    elif args.precomputed_map:
        logger.warning(f"WARNING: --precomputed_map set but {map_file} not found, running MAP")
        args.precomputed_map = False

    if not args.precomputed_map:

        def find_map(init_params):
            param_labels = {
                k: "shear" if k in ("g1", "g2") else "default" for k in init_params
            }
            opt_fn = optax.adam if args.map_optimizer == "adam" else optax.adafactor

            # Phase 0: optimize only u/flux with g1,g2 frozen
            if has_shear and args.n_steps_map_freeze_shear > 0:
                optimizer_freeze = optax.multi_transform(
                    transforms={
                        "shear": optax.set_to_zero(),
                        "default": opt_fn(args.lr_map),
                    },
                    param_labels=param_labels,
                )
                opt_state_freeze = optimizer_freeze.init(init_params)

                def update_step_freeze(carry, xs):
                    params, opt_state = carry
                    loss, grads = jax.value_and_grad(nll)(params)
                    updates, opt_state = optimizer_freeze.update(
                        grads, opt_state, params
                    )
                    params = optax.apply_updates(params, updates)
                    aux = (loss, params["g1"], params["g2"]) if has_shear else (loss,)
                    return (params, opt_state), aux

                (init_params, _), scan_out_f = jax.lax.scan(
                    update_step_freeze,
                    (init_params, opt_state_freeze),
                    length=args.n_steps_map_freeze_shear,
                )

            # Phase 1: joint optimization of all parameters
            if has_shear:
                optimizer = optax.multi_transform(
                    transforms={
                        "shear": opt_fn(args.lr_map * args.lr_map_shear_factor),
                        "default": opt_fn(args.lr_map),
                    },
                    param_labels=param_labels,
                )
            else:
                optimizer = opt_fn(args.lr_map)

            opt_state = optimizer.init(init_params)

            def update_step(carry, xs):
                params, opt_state = carry
                loss, grads = jax.value_and_grad(nll)(params)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                aux = (loss, params["g1"], params["g2"]) if has_shear else (loss,)
                return (params, opt_state), aux

            (params, _), scan_out = jax.lax.scan(
                update_step, (init_params, opt_state), length=args.n_steps_map
            )

            # Concatenate traces
            if has_shear:
                if args.n_steps_map_freeze_shear > 0:
                    losses_f, g1_f, g2_f = scan_out_f
                    losses, g1_trace, g2_trace = scan_out
                    losses = jnp.concatenate([losses_f, losses])
                    g1_trace = jnp.concatenate([g1_f, g1_trace])
                    g2_trace = jnp.concatenate([g2_f, g2_trace])
                else:
                    losses, g1_trace, g2_trace = scan_out
                return params, losses, g1_trace, g2_trace
            else:
                if args.n_steps_map_freeze_shear > 0:
                    (losses_f,) = scan_out_f
                    (losses,) = scan_out
                    losses = jnp.concatenate([losses_f, losses])
                else:
                    (losses,) = scan_out
                return params, losses

        t_map_start = datetime.now()
        map_results = jax.vmap(find_map)(map_init_val)
        if has_shear:
            init_val, map_losses, map_g1_trace, map_g2_trace = map_results
            # Block until MAP is complete so the timing reflects actual GPU work
            init_val["g1"].block_until_ready()
        else:
            init_val, map_losses = map_results
            map_losses.block_until_ready()
        t_map_end = datetime.now()
        map_elapsed = t_map_end - t_map_start
        logger.info(f"MAP elapsed: {map_elapsed}  ({args.n_steps_map_freeze_shear + args.n_steps_map} steps × {args.num_chains} chains)")

        # Print MAP diagnostics
        if has_shear:
            logger.info(f"{init_val['g1'] * g_rescale} {init_val['g2'] * g_rescale}")
            logger.info(f"Initial guess: g1={init_val['g1']*g_rescale}, g2={init_val['g2']*g_rescale}")
        logger.info(f"MAP final loss (per chain): {map_losses[:, -1]}")

        if args.save_plots:
            # Plot MAP convergence: loss and g1,g2 evolution
            n_map_plots = 3 if has_shear else 1
            fig, axes = plt.subplots(1, n_map_plots, figsize=(5 * n_map_plots, 4))
            if n_map_plots == 1:
                axes = [axes]
            total_map_steps = (
                args.n_steps_map_freeze_shear if has_shear else 0
            ) + args.n_steps_map
            steps = jnp.arange(total_map_steps)

            # Loss (log scale, shift to ensure positive values)
            loss_min = jnp.min(map_losses)
            loss_offset = jnp.where(loss_min < 0, jnp.abs(loss_min) + 1.0, 0.0)
            for c in range(map_losses.shape[0]):
                axes[0].plot(
                    steps, map_losses[c] + loss_offset, alpha=0.7, label=f"chain {c}"
                )
            axes[0].set_yscale("log")
            if has_shear and args.n_steps_map_freeze_shear > 0:
                axes[0].axvline(
                    args.n_steps_map_freeze_shear,
                    color="k",
                    ls=":",
                    alpha=0.5,
                    label="unfreeze g",
                )
            axes[0].set_xlabel("MAP step")
            ylabel = (
                "Loss (NLL)" if loss_offset == 0 else f"Loss (NLL + {loss_offset:.1f})"
            )
            axes[0].set_ylabel(ylabel)
            axes[0].set_title("MAP loss")
            axes[0].legend(fontsize=7)

            if has_shear:
                # g1 trace per chain: shape (n_chains, total_map_steps, 1) -> squeeze last dim
                map_g1_phys = (
                    map_g1_trace.squeeze(-1) * g_rescale
                )  # (n_chains, total_steps)
                map_g2_phys = map_g2_trace.squeeze(-1) * g_rescale

                for c in range(map_g1_phys.shape[0]):
                    axes[1].plot(steps, map_g1_phys[c], alpha=0.7, label=f"chain {c}")
                    axes[2].plot(steps, map_g2_phys[c], alpha=0.7, label=f"chain {c}")
                axes[1].axhline(args.g1_true, color="r", ls="--", lw=1, label="true")
                axes[2].axhline(args.g2_true, color="r", ls="--", lw=1, label="true")
                if args.n_steps_map_freeze_shear > 0:
                    axes[1].axvline(
                        args.n_steps_map_freeze_shear, color="k", ls=":", alpha=0.5
                    )
                    axes[2].axvline(
                        args.n_steps_map_freeze_shear, color="k", ls=":", alpha=0.5
                    )
                axes[1].set_xlabel("MAP step")
                axes[1].set_ylabel("g1")
                axes[1].set_title("g1 MAP trace")
                axes[1].legend(fontsize=7)
                axes[2].set_xlabel("MAP step")
                axes[2].set_ylabel("g2")
                axes[2].set_title("g2 MAP trace")
                axes[2].legend(fontsize=7)

            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "map_convergence.png"), dpi=150)
            plt.close(fig)
        if args.save_data:
            np.save(
                os.path.join(out_dir, "radio_map_val.npy"), init_val, allow_pickle=True
            )

    if args.point_estimate:
        if has_shear:
            g1_estimates = init_val["g1"] * g_rescale
            g2_estimates = init_val["g2"] * g_rescale
            np.save(
                os.path.join(out_dir, "map_shear_estimates.npy"),
                jnp.stack([g1_estimates, g2_estimates], axis=-1),
            )
            logger.info(f"Point estimate g1 (per chain): {g1_estimates}")
            logger.info(f"Point estimate g2 (per chain): {g2_estimates}")
            logger.info(f"Point estimate g1 mean: {jnp.mean(g1_estimates):.6f}, g2 mean: {jnp.mean(g2_estimates):.6f}")
            logger.info(f"True values: g1={args.g1_true}, g2={args.g2_true}")
        else:
            logger.info("No shear parameters — point estimate not applicable")
        logger.info(f"Point estimate saved to {out_dir}")
        t_end = datetime.now()
        logger.info(f"End: {t_end.strftime('%Y-%m-%d %H:%M:%S')} (elapsed: {t_end - t_start})")
        sys.exit(0)

    if args.save_plots and has_shear:
        # Plot the initial guess for the shear
        plt.figure()
        plt.scatter(
            init_val_["g1"] * (args.g_prior_scale / args.g_prior_sigma),
            init_val_["g2"] * (args.g_prior_scale / args.g_prior_sigma),
            label="Initial guess",
        )
        plt.scatter(
            init_val["g1"] * (args.g_prior_scale / args.g_prior_sigma),
            init_val["g2"] * (args.g_prior_scale / args.g_prior_sigma),
            label="MAP estimate",
        )
        plt.scatter(args.g1_true, args.g2_true, color="red", label="True shear")
        plt.xlim(
            args.g1_true - 3 * args.g_prior_scale, args.g1_true + 3 * args.g_prior_scale
        )
        plt.ylim(
            args.g2_true - 3 * args.g_prior_scale, args.g2_true + 3 * args.g_prior_scale
        )
        plt.xlabel("g1")
        plt.ylabel("g2")
        plt.title("Initial guess for the shear")
        plt.legend()
        # plt.show()
        plt.savefig(os.path.join(out_dir, "radio_initial_guess.png"))

    # Initialise sampling

    key_warmup, key_sample = jax.random.split(key)

    if args.sampler == "mclmc":
        key_init, key_tune = jax.random.split(key_warmup)
        key_init_chains = jax.random.split(key_init, args.num_chains)

        # Compute dimensionality and gradient at MAP for initial step_size/L.
        # Gradient-norm scaling (sqrt(dim)/grad_norm) keeps the leapfrog energy
        # error in a safe regime: overshooting causes a NaN cascade that
        # collapses step_size to 0 permanently. Undershooting just costs warmup.
        first_chain_init = jax.tree.map(lambda x: x[0], init_val)
        ndim = sum(v.size for v in jax.tree.leaves(first_chain_init))
        DESIRED_ENERGY_VAR = 1e-3
        grad_at_map = jax.grad(log_prob_fn)(first_chain_init)
        grad_norm = jnp.linalg.norm(ravel_pytree(grad_at_map)[0])

        # MAP convergence diagnostic: ‖∇‖ should drop by orders of magnitude
        # between pre-MAP and post-MAP. If reduction is small, MAP didn't
        # converge and the gradient-based step_size formula will underestimate.
        first_chain_pre_map = jax.tree.map(lambda x: x[0], init_val_)
        grad_pre_map_norm = jnp.linalg.norm(
            ravel_pytree(jax.grad(log_prob_fn)(first_chain_pre_map))[0]
        )
        reduction = (
            float(grad_pre_map_norm / grad_norm)
            if float(grad_norm) > 0
            else float("inf")
        )
        logger.info(f"MAP gradient reduction: ‖∇‖ pre-MAP={float(grad_pre_map_norm):.3e}, "
            f"post-MAP={float(grad_norm):.3e}, factor={reduction:.1f}x")

        # Per-group ‖∇‖ at MAP — identifies which parameters did/didn't converge.
        logger.info("Per-group ‖∇‖ at MAP:")
        for k in sorted(grad_at_map.keys()):
            g_flat = grad_at_map[k].ravel()
            line = (
                f"  {k:>12s}: ‖∇‖={float(jnp.linalg.norm(g_flat)):.3e}, "
                f"max|∂|={float(jnp.max(jnp.abs(g_flat))):.3e}"
            )
            logger.info(line)

        # Fallback: when grad_norm is enormous (un-converged MAP or stiff
        # likelihood), the formula collapses step_size to ~0 and the EMA in
        # phase 1 cannot recover within frac_tune1·n_warmup steps. Floor at
        # args.lr_map so adaptation starts from a workable scale.
        formula_step_size = (
            float(jnp.sqrt(ndim) / grad_norm) * (DESIRED_ENERGY_VAR / 1e-2) ** 0.25
        )
        if formula_step_size < args.lr_map:
            initial_step_size = args.lr_map
            init_source = f"floor=lr_map ({args.lr_map:.3e}; formula gave {formula_step_size:.3e})"
        else:
            initial_step_size = formula_step_size
            init_source = f"gradient formula ({formula_step_size:.3e})"
        initial_L = float(jnp.sqrt(ndim)) * initial_step_size
        logger.info(f"MCLMC init: ndim={ndim}, grad_norm={float(grad_norm):.3e}, "
            f"initial_step_size={initial_step_size:.3e}, initial_L={initial_L:.3e} "
            f"[{init_source}]")

        def mclmc_factory(inverse_mass_matrix):
            return blackjax.mcmc.mclmc.build_kernel(
                logdensity_fn=log_prob_fn,
                inverse_mass_matrix=inverse_mass_matrix,
                integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
            )

        # Build inverse mass matrix. JAX flattens parameter dicts in sorted key order,
        # so we iterate sorted keys to match the flattened vector layout.
        # Setting a smaller value for g1/g2 gives them finer effective steps,
        # compensating for their much narrower posterior relative to latent dims.
        if args.mclmc_inv_mass_file is not None:
            inverse_mass_matrix = jnp.array(np.load(args.mclmc_inv_mass_file))
            logger.info(f"MCLMC inverse mass matrix loaded from {args.mclmc_inv_mass_file}")
            logger.info(f"  shape={inverse_mass_matrix.shape}, min={inverse_mass_matrix.min():.6f}, "
                f"max={inverse_mass_matrix.max():.6f}, ratio={inverse_mass_matrix.max()/inverse_mass_matrix.min():.1f}")
            logger.info(f"MCLMC inverse mass matrix loaded from {args.mclmc_inv_mass_file}")
        elif args.mclmc_inv_mass_shear is not None:
            inv_mass_parts = []
            for k in sorted(first_chain_init.keys()):
                val = args.mclmc_inv_mass_shear if k in ("g1", "g2") else 1.0
                inv_mass_parts.append(jnp.full(first_chain_init[k].size, val))
            inverse_mass_matrix = jnp.concatenate(inv_mass_parts)
            logger.info(f"MCLMC diagonal inverse mass matrix: g1/g2={args.mclmc_inv_mass_shear}, others=1.0")
        else:
            inverse_mass_matrix = jnp.ones((ndim,))

        temp_kernel = blackjax.mclmc(
            log_prob_fn,
            step_size=initial_step_size,
            L=initial_L,
            inverse_mass_matrix=inverse_mass_matrix,
        )

        # Skip adaptation if both L and step_size are provided
        first_chain_state = temp_kernel.init(first_chain_init, key_init_chains[0])

        if args.mclmc_L is not None and args.step_size is not None:
            logger.info(f"Skipping MCLMC adaptation: using L={args.mclmc_L}, step_size={args.step_size}")
            parameters = mclmc_adj.MCLMCAdaptationState(
                L=jnp.array(args.mclmc_L),
                step_size=jnp.array(args.step_size),
                inverse_mass_matrix=jnp.array(inverse_mass_matrix),
            )
        else:
            # Build phase-1+2 adaptation factory once. Calling
            # make_L_step_size_adaptation directly (instead of the
            # mclmc_find_L_and_step_size wrapper) lets us inject the
            # gradient-aware initial L/step_size computed above. The wrapper
            # would discard them and start at L=sqrt(dim), step_size=sqrt(dim)*0.25,
            # which for stiff radio likelihoods triggers a NaN cascade and
            # wastes phase-1 steps on step_size_max recovery.
            frac_tune1, frac_tune2, frac_tune3 = 0.4, 0.4, 0.2
            L_step_size_adapt = mclmc_adj.make_L_step_size_adaptation(
                kernel=mclmc_factory,
                dim=ndim,
                frac_tune1=frac_tune1,
                frac_tune2=frac_tune2,
                desired_energy_var=1e-3,
                trust_in_estimate=2.0,
                num_effective_samples=50,
                diagonal_preconditioning=True,
            )

            initial_params = mclmc_adj.MCLMCAdaptationState(
                L=jnp.array(initial_L),
                step_size=jnp.array(initial_step_size),
                inverse_mass_matrix=jnp.array(inverse_mass_matrix),
            )

            max_adapt_attempts = 10
            for adapt_attempt in range(1, max_adapt_attempts + 1):
                logger.info(f"MCLMC adaptation attempt {adapt_attempt}/{max_adapt_attempts}...")
                key_tune, key_retry = jax.random.split(key_tune)
                key_phase12, key_phase3 = jax.random.split(key_retry)

                # Phase 1+2: adapt step_size, L, and inverse_mass_matrix
                adapted_state, parameters = L_step_size_adapt(
                    first_chain_state, initial_params, args.n_warmup, key_phase12
                )
                logger.info(f"  After phase 1+2: L={float(parameters.L):.6f}, "
                    f"step_size={float(parameters.step_size):.8f}")

                # Phase 3: refine L via ESS — only if phase 1+2 didn't collapse.
                # A collapsed phase 1+2 means L or step_size went to ~0 (dead chain),
                # so phase 3 would just propagate the dead state.
                chain_ok = (
                    float(parameters.L) > 1e-10 and float(parameters.step_size) > 1e-10
                )
                if chain_ok and frac_tune3 > 0:
                    adapted_kernel = mclmc_factory(parameters.inverse_mass_matrix)
                    adapted_state, parameters = mclmc_adj.make_adaptation_L(
                        adapted_kernel, frac=frac_tune3, Lfactor=0.4
                    )(adapted_state, parameters, args.n_warmup, key_phase3)
                    logger.info(f"  After phase 3:   L={float(parameters.L):.6f}, "
                        f"step_size={float(parameters.step_size):.8f}")
                elif not chain_ok:
                    logger.info(f"  Phase 1+2 collapsed; skipping phase 3")

                if parameters.step_size > 0 and parameters.L > 0:
                    break
                logger.info(f"Adaptation failed (step_size={parameters.step_size}, L={parameters.L}), retrying...")

            if parameters.step_size <= 0 or parameters.L <= 0:
                msg = (
                    f"MCLMC adaptation failed after {max_adapt_attempts} attempts: "
                    f"step_size={parameters.step_size}, L={parameters.L}"
                )
                logger.info(msg)
                t_end = datetime.now()
                logger.info(f"End: {t_end.strftime('%Y-%m-%d %H:%M:%S')} (elapsed: {t_end - t_start})")
                raise RuntimeError(msg)

            logger.info(f'Step size: {parameters.step_size}')
            logger.info(f"Step size: {parameters.step_size}")
            logger.info(f'L: {parameters.L}')
            logger.info(f"L: {parameters.L}")
            inv_mass = parameters.inverse_mass_matrix
            if hasattr(inv_mass, "shape") and inv_mass.ndim > 0:
                logger.info(f"Inverse mass matrix: min={inv_mass.min():.6f}, max={inv_mass.max():.6f}, "
                    f"median={jnp.median(inv_mass):.6f}, ratio={inv_mass.max()/inv_mass.min():.1f}")
                # Print per-parameter group and save full vector
                offset = 0
                for k in sorted(first_chain_init.keys()):
                    size = first_chain_init[k].size
                    chunk = inv_mass[offset : offset + size]
                    logger.info(f"  {k:>6s} [{size:4d}]: min={chunk.min():.6f}, max={chunk.max():.6f}, median={jnp.median(chunk):.6f}")
                    offset += size
                np.save(
                    os.path.join(out_dir, "mclmc_inv_mass_matrix.npy"),
                    np.array(inv_mass),
                )
                logger.info(f"Saved inverse mass matrix to {out_dir}/mclmc_inv_mass_matrix.npy")
            else:
                logger.info(f"Inverse mass matrix: scalar = {inv_mass}")

        # Convert VAE to float16 for sampling (after adaptation in float32)
        if args.model_profile == "VAE" and args.vae_precision == "float16":
            ae = jax.tree.map(
                lambda x: (
                    x.astype(jnp.float16)
                    if isinstance(x, jnp.ndarray)
                    and jnp.issubdtype(x.dtype, jnp.floating)
                    else x
                ),
                ae,
            )
            jitted_decode = eqx.filter_jit(
                lambda z, key: ae.decode(z.astype(jnp.float16), key=key)
            )
            if args.use_flow:
                model = partial(
                    model_fn_VAE_flow,
                    Ngal=args.Ngal,
                    Npx=args.Npx,
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
                    latent_sigma=args.latent_sigma,
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
                    latent_sigma=args.latent_sigma,
                    jitted_decode=jitted_decode,
                    gsparams=gsparams,
                    run_type=args.vae_model_inference_mode,
                    batch_size=args.vae_inference_batch_size,
                    use_dropout=args.use_dropout,
                )
            seeded_model = seed(model, jax.random.PRNGKey(0))

            @jax.jit
            def log_prob_fn(params):
                @jax.checkpoint
                def _log_density(params):
                    return numpyro.infer.util.log_density(
                        seeded_model,
                        (),
                        {"obs": data},
                        params,
                    )[0]

                return _log_density(params)

            logger.info("VAE decoder converted to float16 for sampling")

        # Build the final kernel with tuned parameters
        kernel = blackjax.mclmc(log_prob_fn, **parameters._asdict())

        # Initialize all chains with tuned kernel
        last_states = jax.vmap(kernel.init)(init_val, key_init_chains)

    else:
        raise ValueError("Sampler not recognized. Use mclmc.")

    # === SAMPLING LOOP ===
    @partial(jax.jit, static_argnames=("num_steps",))
    def run_hmc(init_states, key, num_steps=1):

        def make_step(state, key):
            state, info = kernel.step(key, state)
            return state, (state, info)

        keys = jax.random.split(key, num_steps)
        last_states, (samples, info) = jax.lax.scan(make_step, init_states, keys)

        return last_states, (samples, info)

    # loop over lax.scan to save GPU memory
    logger.info(f"Number of chains: {args.num_chains}")
    logger.info(f"Number of loops: {args.num}")
    logger.info(f"Number of steps: {args.num_steps}")
    logger.info(f"Number of samples per chain: {args.num_steps*args.num*2}")

    key_chains = jax.random.split(key_sample, args.num_chains)

    last_states, _ = jax.vmap(lambda init_states, keys: run_hmc(init_states, keys, 1))(
        last_states, key_chains
    )

    sample_list = []

    keys = jax.vmap(jax.random.split, in_axes=(0, None))(key_chains, 2 * args.num)

    for i in range(args.num):
        logger.info(f'Chain {i + 1} of {2 * args.num} running...')
        last_states, (samples, info) = jax.vmap(
            lambda init_states, keys: run_hmc(init_states, keys, args.num_steps)
        )(last_states, keys[:, i, :])
        sample_list.append(samples)

        # Quick diagnostics: sampler health and shear chain statistics.
        # MCLMC info has no acceptance_rate; energy_change measures integrator accuracy.
        sampler_diag = (
            f"mean|energy_change|={float(jnp.abs(info.energy_change).mean()):.3f}"
        )
        diag = f"  {sampler_diag}"
        if has_shear:
            g1_mean = float(samples.position["g1"].mean()) * g_rescale
            g1_std = float(samples.position["g1"].std()) * g_rescale
            g2_mean = float(samples.position["g2"].mean()) * g_rescale
            g2_std = float(samples.position["g2"].std()) * g_rescale
            diag += (
                f" | g1={g1_mean:.4f}±{g1_std:.4f}" f" | g2={g2_mean:.4f}±{g2_std:.4f}"
            )
        logger.info(diag)

    samples_ = {
        key: np.concatenate([sample_list[k].position[key] for k in range(args.num)], 1)
        for key in last_states.position
    }
    # Print ESS for all parameters
    logger.info("ESS at the end of first loop")
    for k in sorted(samples_.keys()):
        if samples_[k].ndim >= 3 and samples_[k].shape[-1] > 1:
            # Multi-dimensional param (e.g. z, flux per galaxy): report mean ESS over first component
            if samples_[k].ndim == 5:  # z: (chains, steps, Ngal, d1, d2)
                ess_vals = [
                    blackjax.diagnostics.effective_sample_size(
                        samples_[k][:, :, gal, 0, 0]
                    )
                    for gal in range(min(args.Ngal, samples_[k].shape[2]))
                ]
                ess_mean = np.mean(ess_vals)
                logger.info(f"ESS {k} (mean over galaxies, first component) {ess_mean:.1f}")
            else:
                ess = blackjax.diagnostics.effective_sample_size(samples_[k][..., 0])
                logger.info(f"ESS {k} {ess:.1f}")
        else:
            ess = blackjax.diagnostics.effective_sample_size(samples_[k][..., 0])
            logger.info(f"ESS {k} {ess:.1f}")

    # extra chains
    for i in range(args.num):
        logger.info(f'Extra chain {args.num + i + 1} of {2 * args.num} running...')
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

    # Build labels list from available sample keys (generic over model)
    # Plot scalar/low-dim params; skip high-dim latents (z, u) but add first component
    labels = [k for k in sorted(samples_.keys()) if k not in ("z", "u")]
    latent_key = "z" if "z" in samples_ else ("u" if "u" in samples_ else None)
    if latent_key is not None:
        labels.append(f"{latent_key}[0]")

    if args.save_plots:
        fig, axes = plt.subplots(
            max(len(labels), 1), figsize=(10, 2.5 * max(len(labels), 1)), sharex=True
        )
        if len(labels) == 1:
            axes = [axes]
        for i, label in enumerate(labels):
            ax = axes[i]
            for k in range(args.num_chains):
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
        if args.plot_chains in ["samples", "both"]:
            plt.savefig(os.path.join(out_dir, "radio_chains.png"))
        plt.close()

        # Scaled chains plot
        fig, axes = plt.subplots(
            max(len(labels), 1), figsize=(10, 2.5 * max(len(labels), 1)), sharex=True
        )
        if len(labels) == 1:
            axes = [axes]
        for i, label in enumerate(labels):
            ax = axes[i]
            for k in range(args.num_chains):
                if label in ("hlr", "hlr_disk", "hlr_bulge") and label in samples_:
                    ax.plot(
                        jax.nn.sigmoid(samples_[label][k, :, 0] / args.hlr_prior_sigma)
                        * (args.hlr_prior_max - args.hlr_prior_min)
                        + args.hlr_prior_min,
                        "k",
                        alpha=0.3,
                    )
                elif label == "flux":
                    ax.plot(
                        jax.nn.sigmoid(
                            samples_["flux"][k, :, 0] / args.flux_prior_sigma
                        )
                        * (args.flux_prior_max - args.flux_prior_min)
                        + args.flux_prior_min,
                        "k",
                        alpha=0.3,
                    )
                elif label == "flux_ratio":
                    ax.plot(
                        jax.nn.sigmoid(samples_["flux_ratio"][k, :, 0])
                        * args.composite_flux_ratio_max,
                        "k",
                        alpha=0.3,
                    )
                elif label in ["e1", "e2"] and "e1" in samples_ and "e2" in samples_:
                    e = jnp.stack(
                        [
                            samples_["e1"][k, :, 0]
                            / args.ell_prior_sigma
                            * args.ell_prior_scale,
                            samples_["e2"][k, :, 0]
                            / args.ell_prior_sigma
                            * args.ell_prior_scale,
                        ],
                        0,
                    )
                    e = to_unit_disk(e)
                    ax.plot(e[0] if label == "e1" else e[1], "k", alpha=0.3)
                elif label in ["e1_disk", "e2_disk"] and "e1_disk" in samples_:
                    e = jnp.stack(
                        [
                            samples_["e1_disk"][k, :, 0]
                            / args.ell_prior_sigma
                            * args.ell_prior_scale,
                            samples_["e2_disk"][k, :, 0]
                            / args.ell_prior_sigma
                            * args.ell_prior_scale,
                        ],
                        0,
                    )
                    e = to_unit_disk(e)
                    ax.plot(e[0] if label == "e1_disk" else e[1], "k", alpha=0.3)
                elif label in ["e1_bulge", "e2_bulge"] and "e1_bulge" in samples_:
                    e = jnp.stack(
                        [
                            samples_["e1_bulge"][k, :, 0]
                            / args.ell_prior_sigma
                            * args.ell_prior_scale,
                            samples_["e2_bulge"][k, :, 0]
                            / args.ell_prior_sigma
                            * args.ell_prior_scale,
                        ],
                        0,
                    )
                    e = to_unit_disk(e)
                    ax.plot(e[0] if label == "e1_bulge" else e[1], "k", alpha=0.3)
                elif label in ["g1", "g2"] and "g1" in samples_ and "g2" in samples_:
                    g = jnp.stack(
                        [
                            samples_["g1"][k, :, 0]
                            / args.g_prior_sigma
                            * args.g_prior_scale,
                            samples_["g2"][k, :, 0]
                            / args.g_prior_sigma
                            * args.g_prior_scale,
                        ],
                        0,
                    )
                    g = to_unit_disk(g)
                    ax.plot(g[0] if label == "g1" else g[1], "k", alpha=0.3)
                elif latent_key is not None and label == f"{latent_key}[0]":
                    ax.plot(
                        samples_[latent_key][k, :, 0, 0, 0] / args.latent_sigma,
                        "k",
                        alpha=0.3,
                    )
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
        if args.plot_chains in ["scaled", "both"]:
            plt.savefig(os.path.join(out_dir, "radio_chains_scaled.png"))
        plt.close()

        if has_shear:
            two_truths = np.array([args.g1_true, args.g2_true])
            samples_g = np.concatenate([samples_["g1"], samples_["g2"]], -1).reshape(
                (-1, 2)
            ) * (args.g_prior_scale / args.g_prior_sigma)

            two_labels = [r"$\gamma_1$", r"$\gamma_2$"]

            fig = plt.figure(figsize=(7, 7))
            fig = corner.corner(
                samples_g, truths=two_truths, labels=two_labels, fig=fig
            )
            fig.savefig(os.path.join(out_dir, "radio_corner_g.png"))
            plt.close()

    # Final ESS for all parameters
    logger.info("ESS at the end of second loop")
    for k in sorted(samples_.keys()):
        if samples_[k].ndim == 5:  # z/u: (chains, steps, Ngal, d1, d2)
            ess_vals = [
                blackjax.diagnostics.effective_sample_size(samples_[k][:, :, gal, 0, 0])
                for gal in range(min(args.Ngal, samples_[k].shape[2]))
            ]
            ess_mean = np.mean(ess_vals)
            logger.info(f"ESS {k} (mean over galaxies, first component) {ess_mean:.1f}")
        else:
            ess = blackjax.diagnostics.effective_sample_size(samples_[k][..., 0])
            logger.info(f"ESS {k} {ess:.1f}")

    if has_shear:
        flatchain = np.std(samples_["g1"], axis=1) < 1e-4
        logger.info("Flatchains:")
        logger.info(flatchain)
        logger.info("Flatchains:")
        logger.info(flatchain)

        # Compute posterior density estimation shear samples
        g1_scaled = samples_["g1"][np.where(flatchain == False), :]
        g2_scaled = samples_["g2"][np.where(flatchain == False), :]
        samples_g_scaled = (
            np.concatenate([g1_scaled, g2_scaled], -1).reshape((-1, 2))
            / args.g_prior_sigma
            * args.g_prior_scale
        )

        # Fit and save GMM posterior density
        logger.info("Fitting GMM to shear posterior...")
        gmm_params = fit_gmm(samples_g_scaled, n_components=5)
        save_gmm(gmm_params, os.path.join(out_dir, "radio_shear_gmm.npz"))
        logger.info(f"GMM saved with {len(gmm_params['weights'])} components")

        # Save sample-level mean and std (avoids GMM approximation error)
        g_mean = np.mean(samples_g_scaled, axis=0)
        g_std = np.std(samples_g_scaled, axis=0)
        g_cov = np.cov(samples_g_scaled.T)
        np.savez(
            os.path.join(out_dir, "radio_samples_stats.npz"),
            g1_mean=g_mean[0],
            g2_mean=g_mean[1],
            g1_std=g_std[0],
            g2_std=g_std[1],
            cov=g_cov,
        )

        if args.save_plots:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            plot_gmm_contours(
                gmm_params,
                ax=ax,
                true_g=(args.g1_true, args.g2_true),
            )
            ax.set_title("GMM Posterior Density")
            fig.savefig(
                os.path.join(out_dir, "gmm_contours.png"), dpi=150, bbox_inches="tight"
            )
            plt.close(fig)

    # Save samples
    if args.save_samples:
        logger.info("Saving samples...")
        np.savez(os.path.join(out_dir, "radio_samples.npz"), **samples_)

    # Print arguments
    logger.info("Arguments:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")

    # Save log file
    t_end = datetime.now()
    logger.info(f"End: {t_end.strftime('%Y-%m-%d %H:%M:%S')} (elapsed: {t_end - t_start})")

    logger.info("Done.")


if __name__ == "__main__":
    main()
