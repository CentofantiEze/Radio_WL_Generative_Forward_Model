import warnings
from datetime import datetime
from functools import partial
from pathlib import Path

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import optax
import equinox as eqx
from numpyro.handlers import seed, trace

warnings.filterwarnings("ignore")

import json
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.shearest.cli import parse_args
from src.shearest.logging_setup import setup_logger
from src.shearest.data_gen_utils import gen_gal_dataset
from src.shearest.model_utils import (
    setup_vae_state,
    cast_ae_to_float16,
    build_model,
    build_log_prob_fn,
)
from src.shearest.psf_utils import compute_radio_uv_mask
from src.shearest.posterior_utils import fit_gmm, save_gmm
from src.shearest import plotting, sampling

# Data-gen AE (needs encoder + decoder) — different checkpoint from the
# model AE (which is handled inside setup_vae_state / cast_ae_to_float16).
from pshear.utils import load_galaxy_autoencoder  # type: ignore
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
    logger.info(
        f"noise_data: {args.noise_data if args.noise_data is not None else f'{args.noise_uv} (fallback to noise_uv)'}"
    )
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
        plotting.plot_uv_mask_psf(mask, psf, out_dir)

    # Init seed. Use ``np.random.default_rng()`` (modern numpy RNG, seeded
    # from OS entropy via SeedSequence) rather than the legacy global state
    # ``np.random.randint`` so we don't depend on whether some other library
    # has touched the global RNG before us.
    if args.seed is None:
        args.seed = int(np.random.default_rng().integers(1, 1_000_000))
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
            logger.info(
                f"Loading data-generation AE from {data_vae_path} epoch {data_vae_epoch}"
            )
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
    # Load the model AE + flow (no-op when profile is not VAE) and build the
    # numpyro forward-model partial. The AE starts in float32 for MAP stability;
    # it is cast to float16 later, after MCLMC adaptation.
    ae = jitted_decode = gsparams = flow_forward = flow_condition = None
    if args.model_profile == "VAE":
        ae, jitted_decode, gsparams, flow_forward, flow_condition = setup_vae_state(
            args, logger
        )
    model = build_model(
        args,
        data=data,
        uv_pos=uv_pos,
        jitted_decode=jitted_decode,
        gsparams=gsparams,
        flow_forward=flow_forward,
        flow_condition=flow_condition,
    )

    if args.save_plots:
        uv_images = plotting.plot_data_grid(data, mask, uv_pos, args.Ngal, out_dir)
        plotting.plot_random_galaxy(uv_images, args.Ngal, out_dir)

    # Sample parameters from their prior.
    # ``name`` is the trace-key (parameter name, e.g. "g1", "u", "flux");
    # kept distinct from ``key`` (the PRNG key) to avoid the previous shadowing.
    def draw_params(key):
        t = trace(seed(model, key)).get_trace()
        return {name: t[name]["value"] for name in t if name != "obs"}

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
        logger.info(
            f"g_chains_init: g1={g1_init}, g2={g2_init} (physical) -> g1={g1_mcmc:.4f}, g2={g2_mcmc:.4f} (MCMC space)"
        )

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
        logger.info(
            f"u_chains_init: u ~ N(0, {args.u_chains_init}^2) (shape {u_shape})"
        )

    if args.save_data:
        np.save(
            os.path.join(out_dir, "radio_init_val.npy"), init_val_, allow_pickle=True
        )

    # Build the (jitted, gradient-checkpointed) log density for the joint
    # distribution conditioned on `data`. The model is seeded with a fixed
    # key inside build_log_prob_fn so the density is deterministic — required
    # for gradient-based MCMC.
    log_prob_fn = build_log_prob_fn(model, data)

    logger.info(
        f"MAP optimizer: {args.map_optimizer}, lr: {args.lr_map} (shear factor: {args.lr_map_shear_factor}x)"
    )
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
            logger.info(
                f"Loaded MAP: g1={init_val['g1']*g_rescale}, g2={init_val['g2']*g_rescale}"
            )
    elif args.precomputed_map:
        logger.warning(
            f"WARNING: --precomputed_map set but {map_file} not found, running MAP"
        )
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
        logger.info(
            f"MAP elapsed: {map_elapsed}  ({args.n_steps_map_freeze_shear + args.n_steps_map} steps × {args.num_chains} chains)"
        )

        # Print MAP diagnostics
        if has_shear:
            logger.info(
                f"Initial guess: g1={init_val['g1']*g_rescale}, g2={init_val['g2']*g_rescale}"
            )
        logger.info(f"MAP final loss (per chain): {map_losses[:, -1]}")

        if args.save_plots:
            plotting.plot_map_convergence(
                map_losses,
                map_g1_trace if has_shear else None,
                map_g2_trace if has_shear else None,
                args,
                has_shear=has_shear,
                g_rescale=g_rescale,
                out_dir=out_dir,
            )
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
            logger.info(
                f"Point estimate g1 mean: {jnp.mean(g1_estimates):.6f}, g2 mean: {jnp.mean(g2_estimates):.6f}"
            )
            logger.info(f"True values: g1={args.g1_true}, g2={args.g2_true}")
        else:
            logger.info("No shear parameters — point estimate not applicable")
        logger.info(f"Point estimate saved to {out_dir}")
        t_end = datetime.now()
        logger.info(
            f"End: {t_end.strftime('%Y-%m-%d %H:%M:%S')} (elapsed: {t_end - t_start})"
        )
        sys.exit(0)

    if args.save_plots and has_shear:
        plotting.plot_initial_guess_shear(init_val_, init_val, args, out_dir)

    # Initialise sampling

    key_warmup, key_sample = jax.random.split(key)

    if args.sampler == "mclmc":
        # Initial step_size/L from MAP gradient, inverse-mass-matrix setup,
        # phase 1+2 + phase 3 adaptation with retries, and the diagnostic
        # save of the adapted mass matrix all live in :mod:`sampling`.
        parameters, key_init_chains = sampling.setup_mclmc(
            log_prob_fn,
            init_val_map=init_val,
            init_val_prior=init_val_,
            key_warmup=key_warmup,
            args=args,
            out_dir=out_dir,
            logger=logger,
        )

        # Cast the decoder to float16 for sampling (after adaptation in float32)
        # and rebuild the model + log-density on top of the f16 decoder.
        if args.model_profile == "VAE" and args.vae_precision == "float16":
            ae, jitted_decode = cast_ae_to_float16(ae)
            model = build_model(
                args,
                data=data,
                uv_pos=uv_pos,
                jitted_decode=jitted_decode,
                gsparams=gsparams,
                flow_forward=flow_forward,
                flow_condition=flow_condition,
            )
            log_prob_fn = build_log_prob_fn(model, data)

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
        logger.info(f"Chain {i + 1} of {2 * args.num} running...")
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
                logger.info(
                    f"ESS {k} (mean over galaxies, first component) {ess_mean:.1f}"
                )
            else:
                ess = blackjax.diagnostics.effective_sample_size(samples_[k][..., 0])
                logger.info(f"ESS {k} {ess:.1f}")
        else:
            ess = blackjax.diagnostics.effective_sample_size(samples_[k][..., 0])
            logger.info(f"ESS {k} {ess:.1f}")

    # extra chains
    for i in range(args.num):
        logger.info(f"Extra chain {args.num + i + 1} of {2 * args.num} running...")
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
        plotting.plot_chains_raw(
            samples_,
            labels,
            latent_key,
            args.num_chains,
            args.plot_chains,
            out_dir,
        )
        plotting.plot_chains_scaled(samples_, labels, latent_key, args, out_dir)
        if has_shear:
            plotting.plot_corner_shear(samples_, args, out_dir)

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
            plotting.plot_gmm_posterior(gmm_params, args, out_dir)

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
    logger.info(
        f"End: {t_end.strftime('%Y-%m-%d %H:%M:%S')} (elapsed: {t_end - t_start})"
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
