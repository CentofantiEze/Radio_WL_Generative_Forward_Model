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
    model_utils.py     # numpyro model_fn{_VAE, _VAE_flow, _composite, …}
    data_gen_utils.py  # draw_{exp,spergel,HST,AE_HST,NN}_profile  + gen_gal_dataset
    psf_utils.py       # compute_radio_uv_mask via argosim
    func_utils.py      # to_unit_disk, complex/stack helpers
    posterior_utils.py # GMM fitting and combination

scripts/
    shear_numpyro_sampling_argparse.py   # main entry point

notebooks/             # diagnostic / exploration notebooks
outputs/
    papers/figs/       # paper-ready figures
    papers/            # paper-oriented analysis notebooks
data/
    SKA-Mid.txt        # SKA-Mid antenna positions
    trecs_gal_params.npy
```

## Dependencies

The pipeline requires the following packages: JAX, numpyro, blackjax, jax-galsim, equinox, argosim, optax, corner, flowjax and the **pshear** package providing the autoencoder and flow.


## Usage

The main entry point is `scripts/shear_numpyro_sampling_argparse.py`. All variants of the pipeline are configured via command-line arguments; convenience shell scripts wrap common configurations. You can find some example shell scripts with different configurations in `scripts/`.

## Pipeline structure

The main script follows this sequence:

1. **PSF generation** (`compute_radio_uv_mask` in `psf_utils.py`) — antenna config (SKA-Mid file or random) → UV coverage mask → dirty PSF.
2. **Data generation** (`gen_gal_dataset` in `data_gen_utils.py`) — draws galaxies (parametric Spergel, raw HST cutouts, or AE-whitened COSMOS via `draw_AE_HST_profiles`) → renders k-image → samples at `uv_pos` → adds Gaussian noise at `noise_data`.
3. **MAP estimation** — Samplin initialisation. Adam optimisation on the joint negative log-posterior with multi-transform learning rates (`lr_map` for nuisance, `lr_map · lr_map_shear_factor` for `γ`). Multiple chains in parallel.
4. **MCLMC adaptation** — direct call to `make_L_step_size_adaptation` + `make_adaptation_L` with gradient-aware initial `step_size` and diagonal mass-matrix preconditioning.
5. **Sampling** — MCLMC is run using the `blackjax` implementation.
6. **Posterior summarisation** — fit a 5-component GMM in `(γ₁, γ₂)`, save `radio_shear_gmm.npz`.

The whole pipeline runs end-to-end in ~50 min on a V100 for 100 galaxies with the AE+flow model (`n_warmup=5000, num_chains=4, num_steps=500, num=5`), reaching ESS ≈ 200 on the shear parameters. The parametric Spergel model is faster (~10 min) since the galaxy generative model is much simpler.

<!-- ## References

- **pshear** (autoencoder + flow training and helpers): https://github.com/CentofantiEze/probabilistic-shear
- **jax-galsim**: https://github.com/GalSim-developers/JAX-GalSim
- **blackjax MCLMC**: https://github.com/blackjax-devs/blackjax · Robnik et al. (2023), [arXiv:2212.08549](https://arxiv.org/abs/2212.08549)
- **argosim** (radio array simulation): used to build the UV mask
- **COSMOS 25.2 sample** for galaxy stamps -->

---

⚠️ This repository is under active development
