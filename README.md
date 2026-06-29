# Radio WL Generative Forward Model

Differentiable generative forward model and Bayesian shear inference pipeline for **radio weak lensing**. Built on JAX, [`jax-galsim`](https://github.com/GalSim-developers/JAX-GalSim), [`numpyro`](https://github.com/pyro-ppl/numpyro), and [`blackjax`](https://github.com/blackjax-devs/blackjax) (MCLMC sampler), with a normalising-flow-reparameterised autoencoder prior trained on COSMOS HST stamps.

## Overview

The pipeline simulates radio interferometric observations of galaxies, samples the joint posterior over cosmic shear `(γ₁, γ₂)` and per-galaxy nuisance parameters via gradient-based MCMC, and produces per-run GMM/Gaussian posterior summaries that can be combined across runs.

Two forward models are supported:

1. **Parametric** — Spergel / Exponential / composite (bulge + disk) profiles with per-galaxy `(e₁, e₂, h_l_r, flux, ν)` priors. No machine learning involved.
2. **AE + Flow generative** — galaxies are represented by latent codes `u ∼ 𝒩(0, I)`, transformed through an **unconditional normalising flow** to a learned VAE latent `z`, then decoded by a pretrained **galaxy autoencoder**. Trained on COSMOS stamps so the prior matches the morphological distribution of real galaxies.

The AE encoder/decoder and the latent-space normalising flow live in the companion repository [**pshear** (`probabilistic-shear`)](https://github.com/b-remy/probabilistic-shear) — clone and install that one first.

## Main results

Shear posterior on 10000 AE-whitened COSMOS galaxies observed with SKA-Mid (8 h track, 1.4 GHz, σ_uv = 0.01):

![Combined shear estimates: Flow+AE vs Spergel](outputs/papers/figs/cosmos_shear_estimates.png)

- **Spergel** (purple): 68 / 95 / 99.7 % credible regions of the shear posterior, using the parametric forward model. Recovers a tight but **biased** posterior (~10 σ away from the truth on both `γ₁` and `γ₂`) — the parametric prior is too restrictive to capture COSMOS morphologies, so the inferred shear is model biased.
- **Flow + AE** (orange): 68 / 95 / 99.7 % credible regions of the shear posterior, using the generative forward model. Recovers a **correctly centred** posterior with slightly larger uncertainty than the Spergel case, but without the bias (within 2 σ).

The AE+flow result is reproduced by `outputs/papers/notebooks/cosmos_shear_estimates_plot.ipynb` from the precomputed `combined_gmm_results.npz` summaries in each subdirectory.

## Repository layout

```
src/shearest/          # core library
    cli.py             # parse and validate CLI args
    logging_setup.py   # setup_logger (stdout + timestamped file handler)
    model_utils.py     # numpyro forward models
    data_gen_utils.py  # galaxy generative models
    psf_utils.py       # compute radio PSF via argosim
    sampling.py        # MCLMC setup and adaptation
    plotting.py        # plotting utilities 
    func_utils.py      # helper functions
    posterior_utils.py # GMM fitting and combination

scripts/
    run.py             # main entry point

notebooks/             # diagnostic / exploration notebooks

outputs/
    papers/figs/       # paper-ready figures
    papers/jobs/       # bash scripts for paper runs
data/
    SKA-Mid.txt        # SKA-Mid antenna positions
    trecs_gal_params.npy
```

## Dependencies

The pipeline requires the following packages: JAX, numpyro, blackjax, jax-galsim, equinox, argosim, optax, corner, flowjax and the **pshear** package providing the autoencoder and flow.


## Usage

The main entry point is `scripts/run.py`. All variants of the pipeline are configured via command-line arguments (parsed by `src/shearest/cli.py::parse_args`, which also performs cross-argument validation); convenience shell scripts wrap common configurations. You can find some example shell scripts with different configurations in `outputs/papers/jobs`.

## Pipeline structure

`scripts/run.py` follows this sequence:

1. **CLI parsing & logging** (`src/shearest/cli.py`, `src/shearest/logging_setup.py`) — parse and validate CLI args (`parse_args`), create the output directory, configure the package logger (stdout + a timestamped file handler under `out_dir/`).
2. **PSF generation** (`compute_radio_uv_mask` in `psf_utils.py`) — antenna config (SKA-Mid file or random) → UV coverage mask → dirty PSF.
3. **Data generation** (`gen_gal_dataset` in `data_gen_utils.py`) — draws galaxies (parametric Spergel, raw HST cutouts, or AE-whitened COSMOS via `draw_AE_HST_profiles`) → renders k-image → samples at `uv_pos` → adds Gaussian noise at `noise_data`.
4. **Forward-model build** (`setup_vae_state` + `build_model` + `build_log_prob_fn` in `model_utils.py`) — loads the model AE and (optional) latent normalising flow, builds the numpyro model (`model_fn`, `model_fn_VAE`, `model_fn_VAE_flow`, `model_fn_composite`, …), and wraps the `seed → jit → checkpoint → log_density` pipeline into a single log-probability function for MCMC.
5. **MAP estimation** — sampler initialisation. Adam or Adafactor (selected via `--map_optimizer`) on the joint negative log-posterior with multi-transform learning rates (`lr_map` for nuisance parameters, `lr_map · lr_map_shear_factor` for `γ`). Multiple chains in parallel. Result is cached as `radio_map_val.npy` and re-used on subsequent runs when `--precomputed_map` is set.
6. **MCLMC adaptation** (`sampling.setup_mclmc` in `sampling.py`) — gradient-aware initial `step_size` and `L` from the MAP gradient, diagonal mass-matrix preconditioning, direct call to `make_L_step_size_adaptation` + `make_adaptation_L`, with automatic retries on collapse.
7. **Float16 decoder cast** (`cast_ae_to_float16` in `model_utils.py`) — the VAE decoder is rebuilt in `float16` for the sampling loop (MAP and adaptation run in `float32` for stability); ~2× speedup on V100.
8. **Sampling** — MCLMC run via the `blackjax` implementation, in two outer iterations of `num_chains × num_steps` per chain (`jax.lax.scan` inside, vmapped over chains).
9. **Posterior summarisation** (`fit_gmm` / `save_gmm` in `posterior_utils.py`) — fit a 5-component GMM in `(γ₁, γ₂)`, save `radio_shear_gmm.npz`, and also save sample-level mean and standard deviation.
10. **Plotting** (`plotting.py`) — when `--save_plots` is set: UV mask / PSF, data grid, MAP convergence, chain traces (raw and scaled), corner plot, GMM posterior overlay.

The whole pipeline runs end-to-end in ~50 min on a V100 for 100 galaxies with the AE+flow model (`n_warmup=5000, num_chains=4, num_steps=500, num=5`), reaching ESS ≈ 200 on the shear parameters. The parametric Spergel model is faster (~10 min) since the galaxy generative model is much simpler.

<!-- ## References

- **pshear** (autoencoder + flow training and helpers): https://github.com/CentofantiEze/probabilistic-shear
- **jax-galsim**: https://github.com/GalSim-developers/JAX-GalSim
- **blackjax MCLMC**: https://github.com/blackjax-devs/blackjax · Robnik et al. (2023), [arXiv:2212.08549](https://arxiv.org/abs/2212.08549)
- **argosim** (radio array simulation): used to build the UV mask
- **COSMOS 25.2 sample** for galaxy stamps -->

---

⚠️ This repository is under active development
