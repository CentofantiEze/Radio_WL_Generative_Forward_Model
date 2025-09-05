import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import condition , seed , trace

from einops import rearrange

import optax
from tqdm import tqdm

import blackjax
from functools import partial

import warnings
warnings.filterwarnings('ignore')

import jax_galsim as galsim
import galsim as gs

from psf_utils import compute_radio_uv_mask
from model_utils import model_fn
from data_gen_utils import gen_sersic_profile
from func_utils import stack_2_complex, clip_by_l2_norm

import corner

import argparse

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
    parser.add_argument('--Ngal', type=int, default=100, help='Number of galaxies')
    parser.add_argument('--Npx', type=int, default=128, help='Image size in pixels')
    parser.add_argument('--pixel_scale', type=float, default=0.15, help='Pixel scale in arcsec/pixel')
    parser.add_argument('--noise_uv', type=float, default=0.004, help='UV noise level')
    parser.add_argument('--params_dir', type=str, default='../data/trecs_gal_params.npy', help='Directory for galaxy hlr and flux parameter fit')
    parser.add_argument('--g1_true', type=float, default=-0.05, help='True g1 shear value')
    parser.add_argument('--g2_true', type=float, default=0.05, help='True g2 shear value')
    parser.add_argument('--ell_sigma', type=float, default=0.5, help='Ellipticity prior sigma')
    parser.add_argument('--ell_scale', type=float, default=0.3, help='Ellipticity prior scale')
    parser.add_argument('--g_sigma', type=float, default=1.0, help='Shear prior sigma')
    parser.add_argument('--g_scale', type=float, default=0.3, help='Shear prior scale')
    parser.add_argument('--sersic_index', type=float, default=1.0, help='Sersic index')
    parser.add_argument('--n_antenna', type=int, default=50, help='Number of antennas')
    parser.add_argument('--E_lim', type=float, default=50e3, help='East limit')
    parser.add_argument('--N_lim', type=float, default=50e3, help='North limit')
    parser.add_argument('--track_time', type=float, default=10, help='Track time')
    parser.add_argument('--n_times', type=int, default=4, help='Number of times')
    parser.add_argument('--f', type=float, default=1.4e9, help='Frequency')
    parser.add_argument('--df', type=float, default=1e8, help='Frequency bandwidth')
    parser.add_argument('--n_freqs', type=int, default=4, help='Number of frequency channels')
    parser.add_argument('--radio_array_seed', type=int, default=123, help='Random seed for the radio array generation')
    parser.add_argument('--ell_prior_sigma', type=float, default=0.5, help='Ellipticity prior sigma')
    parser.add_argument('--ell_prior_scale', type=float, default=0.3, help='Ellipticity prior scale')
    parser.add_argument('--g_prior_sigma', type=float, default=1.0, help='Shear prior sigma')
    parser.add_argument('--g_prior_scale', type=float, default=0.3, help='Shear prior scale')
    parser.add_argument('--hlr_prior_sigma', type=float, default=2.0, help='Half-light radius prior sigma')
    parser.add_argument('--hlr_prior_offset', type=float, default=1.0, help='Half-light radius prior offset')
    parser.add_argument('--hlr_prior_scale', type=float, default=0.7142857142857143, help='Half-light radius prior scale')
    parser.add_argument('--hlr_prior_min', type=float, default=0.2, help='Half-light radius prior min')
    parser.add_argument('--flux_prior_sigma', type=float, default=2.0, help='Flux prior sigma')
    parser.add_argument('--flux_prior_offset', type=float, default=0.0, help='Flux prior offset')
    parser.add_argument('--flux_prior_scale', type=float, default=0.06666666666666667, help='Flux prior scale')
    parser.add_argument('--flux_prior_min', type=float, default=0.05, help='Flux prior min')
    parser.add_argument('--lr_map', type=float, default=3e-3, help='MAP learning rate')
    parser.add_argument('--n_steps_map', type=int, default=5000, help='Number of steps for MAP')
    parser.add_argument('--n_warmup', type=int, default=500, help='Number of warmup steps for MEADS')
    parser.add_argument('--num_chains', type=int, default=10, help='Number of chains for HMC')
    parser.add_argument('--step_size', type=float, default=0.005, help='Step size for HMC')
    parser.add_argument('--num', type=int, default=20, help='Number of batch iterations')
    parser.add_argument('--num_steps', type=int, default=10000, help='Number of steps for sampling')
    parser.add_argument('--save_samples', type=bool, default=False, help='Whether to save samples')
    parser.add_argument('--seed', type=int, default=33, help='Random seed')

    args = parser.parse_args()
    fov_size = args.Npx * args.pixel_scale / 3600 # in degrees

    # create log file
    log_file = open("../outputs/radio_sampling.log", "w")

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
    uv_pos, mask, psf = compute_radio_uv_mask(n_antenna=args.n_antenna, 
                                              E_lim=args.E_lim, 
                                              N_lim=args.N_lim, 
                                              Npx=args.Npx, 
                                              fov_size=fov_size, 
                                              track_time=args.track_time,
                                              n_times=args.n_times,
                                              f=args.f,
                                              df=args.df,
                                              n_freqs=args.n_freqs,
                                              seed=args.radio_array_seed)

    plt.subplots(1,3, figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(np.real(mask))
    plt.title('UV mask')
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(psf)
    plt.title('Radio PSF')
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(galsim.Gaussian(flux=1., sigma=.2).drawImage(nx=args.Npx, ny=args.Npx, scale=args.pixel_scale).array)
    plt.title('Gaussian PSF')
    plt.colorbar()
    plt.savefig("../outputs/radio_psf.pdf")

    # Init seed
    if args.seed is None:
        args.seed = np.random.randint(1, 1e6)
    print(f"Random seed: {args.seed}")
    print(f"Random seed: {args.seed}", file=log_file)
    key = jax.random.PRNGKey(args.seed)

    # Generate observations
    model_data_gen = partial(gen_sersic_profile,
                            Ngal=args.Ngal, 
                            Npx=args.Npx, 
                            pixel_scale=args.pixel_scale,  
                            uv_pos=uv_pos, 
                            noise_uv=args.noise_uv, 
                            params_dir=args.params_dir,
                            ell_sigma=args.ell_sigma,
                            ell_scale=args.ell_scale,
                            g_sigma=args.g_sigma,
                            g_scale=args.g_scale,
                            n=args.sersic_index) 
    seeded_model_data_gen = seed(model_data_gen, key)
    # Conditioning model to generate observation with [g1, g2]
    conditionned_model = condition(seeded_model_data_gen, {"g1":args.g1_true*jnp.ones((1,))/(args.g_scale/args.g_sigma), "g2":args.g2_true*jnp.ones((1,))/(args.g_scale/args.g_sigma)})
    data = conditionned_model()

    # Init model for sampling
    key, subkey = jax.random.split(key)
    model = partial(model_fn, 
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
                    hlr_offset=args.hlr_prior_offset,
                    hlr_scale=args.hlr_prior_scale,
                    hlr_min=args.hlr_prior_min,
                    flux_sigma=args.flux_prior_sigma,
                    flux_offset=args.flux_prior_offset,
                    flux_scale=args.flux_prior_scale,
                    flux_min=args.flux_prior_min)
    # seeded_model = seed(model, subkey)

    # Save the data
    np.save("../outputs/radio_data.npy", data)
    np.save("../outputs/radio_psf_mask.npy", mask)

    # Plot observations
    data_complex = []
    for vis in stack_2_complex(data, batch=True):
        img_aux = np.zeros_like(mask)
        img_aux[uv_pos] = vis
        data_complex.append(img_aux)
    data_ = rearrange(data_complex, "(n1 n2) h w -> (n1 h) (n2 w)", n1=10, n2=10)
    plt.figure(figsize=(10,10))
    plt.imshow(np.abs(data_), vmin=np.min(np.abs(data_)), vmax=np.max(np.abs(data_)))
    print('Data shape:', data_.shape)
    print('Data max:', np.max(np.abs(data_)))
    print(f'Data max: {np.max(np.abs(data_))}', file=log_file)
    plt.colorbar()
    plt.savefig("../outputs/radio_data.pdf")

    # Plot a random galaxy
    plt.subplots(1,2,figsize=(12,4))
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
    plt.savefig("../outputs/radio_data_galaxy.pdf")

    # Sample parameters from their prior
    def draw_params(key):
        t = trace(seed(model, key)).get_trace()
        return {key:t[key]["value"] for key in t if not key=="obs"}

    keys = jax.random.split(key, args.num_chains)[:args.num_chains]
    init_val_ = jax.vmap(draw_params)(keys)
    np.save("../outputs/radio_init_val.npy", init_val_, allow_pickle=True)

    # Get the log prob of the joint distribution, conditioned on data
    @jax.jit
    def log_prob_fn(params):
        return numpyro.infer.util.log_density(model, (), {"obs":data,}, params)[0]

    print(f"MAP learning rate: {args.lr_map}", file=log_file)
    print(f"MAP number of steps: {args.n_steps_map}", file=log_file)

    # find the MAP for chain initialization
    nll = lambda params: -log_prob_fn(params)

    def find_map(init_params):
        start_learning_rate = args.lr_map
        optimizer = optax.adafactor(start_learning_rate)

        opt_state = optimizer.init(init_params)

        # A simple update loop.
        def update_step(carry, xs):
            params, opt_state = carry
            grads = jax.grad(nll)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), 0.

        (params, _) , _ = jax.lax.scan(update_step, (init_params, opt_state), length=args.n_steps_map)

        return params

    init_val = jax.vmap(find_map)(init_val_)

    print(init_val["g1"]*(args.g_scale/args.g_sigma), init_val["g2"]*(args.g_scale/args.g_sigma))
    print(f"Initial guess: g1={init_val['g1']*(args.g_scale/args.g_sigma)}, g2={init_val['g2']*(args.g_scale/args.g_sigma)}", file=log_file)
    np.save("../outputs/radio_map_val.npy", init_val, allow_pickle=True)

    # Plot the initial guess for the shear
    plt.figure()
    plt.scatter(init_val_["g1"]*(args.g_scale/args.g_sigma), init_val_["g2"]*(args.g_scale/args.g_sigma), label='Initial guess')
    plt.scatter(init_val["g1"]*(args.g_scale/args.g_sigma), init_val["g2"]*(args.g_scale/args.g_sigma), label='MAP estimate')
    plt.scatter(args.g1_true, args.g2_true, color='red', label='True shear')
    plt.xlabel('g1')
    plt.ylabel('g2')
    plt.title('Initial guess for the shear')
    plt.legend()
    # plt.show()
    plt.savefig("../outputs/radio_initial_guess.pdf")

    # Use the the MEADS algorithm for parallel chains on GPUs
    """
    - https://proceedings.mlr.press/v151/hoffman22a/hoffman22a.pdf
    - https://blackjax-devs.github.io/blackjax/autoapi/blackjax/adaptation/meads_adaptation/index.html
    - https://blackjax-devs.github.io/blackjax/autoapi/blackjax/mcmc/ghmc/index.html
    """

    warmup = blackjax.meads_adaptation(
                log_prob_fn,
                num_chains=args.num_chains,
            )

    key_warmup, key_sample = jax.random.split(key)

    (last_states, parameters), _ = warmup.run(
                key_warmup,
                init_val,
                num_steps=args.n_warmup,
            )

    print('Step size:',parameters["step_size"])
    print(f"Step size: {parameters['step_size']}", file=log_file)
    parameters["step_size"] = args.step_size
    print('Set step size to:',parameters["step_size"])
    print(f"Set step size to: {parameters['step_size']}", file=log_file)
    print(parameters.keys(), file=log_file)
    print(parameters, file=log_file)
    np.save("../outputs/radio_meads_warmup.npy", last_states.position, allow_pickle=True)

    kernel = blackjax.ghmc(log_prob_fn, **parameters)

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

    last_states, _ = jax.vmap(lambda init_states, keys: run_hmc(init_states, keys, 1))(last_states, key_chains)

    sample_list = []

    keys = jax.vmap(jax.random.split, in_axes=(0,None))(key_chains,args.num)

    for i in range(args.num):
        print("Chain", i+1, "of", 2*args.num, "running...")
        last_states, (samples, info) = jax.vmap(lambda init_states, keys: run_hmc(init_states, keys, args.num_steps))(last_states, keys[:,i,:])
        sample_list.append(samples)

    samples_ = {key: np.concatenate([sample_list[k].position[key] for k in range(args.num)], 1) for key in last_states.position}
    print("ESS g1", blackjax.diagnostics.effective_sample_size(samples_["g1"][...,0]))
    print("ESS g2", blackjax.diagnostics.effective_sample_size(samples_["g2"][...,0]))
    print("ESS hlr", blackjax.diagnostics.effective_sample_size(samples_["hlr"][...,0]))
    print("ESS flux", blackjax.diagnostics.effective_sample_size(samples_["flux"][...,0]))
    print("ESS e1", blackjax.diagnostics.effective_sample_size(samples_["e1"][...,0]))
    print("ESS e2", blackjax.diagnostics.effective_sample_size(samples_["e2"][...,0]))
    print("ESS at the end of first loop", file=log_file)
    print("ESS g1", blackjax.diagnostics.effective_sample_size(samples_["g1"][...,0]), file=log_file)
    print("ESS g2", blackjax.diagnostics.effective_sample_size(samples_["g2"][...,0]), file=log_file)
    print("ESS hlr", blackjax.diagnostics.effective_sample_size(samples_["hlr"][...,0]), file=log_file)
    print("ESS flux", blackjax.diagnostics.effective_sample_size(samples_["flux"][...,0]), file=log_file)
    print("ESS e1", blackjax.diagnostics.effective_sample_size(samples_["e1"][...,0]), file=log_file)
    print("ESS e2", blackjax.diagnostics.effective_sample_size(samples_["e2"][...,0]), file=log_file)

    # extra chains
    for i in range(args.num):
        print("Extra chain", args.num+i+1, "of", 2*args.num, "running...")
        last_states, (samples, info) = jax.vmap(lambda init_states, keys: run_hmc(init_states, keys, args.num_steps))(last_states, keys[:,i,:])
        sample_list.append(samples)

    # concatenates chains
    samples_ = {key: np.concatenate([sample_list[k].position[key] for k in range(args.num*2)], 1) for key in last_states.position}

    # labels = ["hlr", "flux", "r_ell", "angle_ell", "g1", "g2"]
    labels = ["hlr", "flux", "e1", "e2", "g1", "g2"]
    params_scales = np.load(args.params_dir, allow_pickle=True)[()]

    fig, axes = plt.subplots(len(labels), figsize=(10, 7), sharex=True)
    for i, label in enumerate(labels):
        print(i, label)
        ax = axes[i]
        for k in range(args.num_chains):
            # if label in ["g1", "g2"]:
            #     ax.plot(samples_[label][k,:,0]*0.1, "k", alpha=0.3)
            # else:
            #     ax.plot(samples_[label][k,:,0], "k", alpha=0.3)
            ax.plot(samples_[label][k,:,0], "k", alpha=0.3)
        ax.set_xlim(0, args.num_steps*args.num*2)
        ax.set_ylabel(label)
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    plt.savefig("../outputs/radio_chains.pdf")

    fig, axes = plt.subplots(len(labels), figsize=(10, 7), sharex=True)
    for i, label in enumerate(labels):
        ax = axes[i]
        for k in range(args.num_chains):
            if label == "hlr":
                # hlr -> jax.nn.softplus(hlr + hlr_offset) * hlr_scale + hlr_min
                ax.plot(jax.nn.softplus(samples_["hlr"][k,:,0]/args.hlr_prior_sigma+args.hlr_prior_offset)*args.hlr_prior_scale + args.hlr_prior_min, "k", alpha=0.3)
            if label == "flux":
                # flux -> jax.nn.softplus(flux + flux_offset) * flux_scale + flux_min
                ax.plot(jax.nn.softplus(samples_["flux"][k,:,0]/args.flux_prior_sigma+args.flux_prior_offset)*args.flux_prior_scale + args.flux_prior_min, "k", alpha=0.3)
            if label in ["e1", "e2"]:
                #  e1, e2 -> clip_by_l2_norm
                e = jnp.stack([samples_["e1"][k,:,0]/args.ell_sigma * args.ell_scale, samples_["e2"][k,:,0]/args.ell_sigma * args.ell_scale], 0)
                e = clip_by_l2_norm(e)
                if label == "e1":
                    ax.plot(e[0], "k", alpha=0.3)
                else:
                    ax.plot(e[1], "k", alpha=0.3)
            if label in ["g1", "g2"]:
                # g1, g2 -> clip_by_l2_norm
                g = jnp.stack([samples_["g1"][k,:,0]/args.g_sigma * args.g_scale, samples_["g2"][k,:,0]/args.g_sigma * args.g_scale], 0)
                g = clip_by_l2_norm(g)
                if label == "g1":
                    ax.plot(g[0], "k", alpha=0.3)
                else:
                    ax.plot(g[1], "k", alpha=0.3)
            else:
                pass

        ax.set_xlim(0, args.num_steps*args.num*2)
        ax.set_ylabel(label)
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    plt.savefig("../outputs/radio_chains_scaled.pdf")

    two_truths = np.array([args.g1_true, args.g2_true])
    samples_g = np.concatenate([samples_["g1"], samples_["g2"]], -1).reshape((-1,2)) * (args.g_scale/args.g_sigma)

    two_cols = ["g_1", "g_2"]
    two_labels = [r"$\gamma_1$", r"$\gamma_2$"]

    fig = plt.figure(figsize=(7, 7))
    fig = corner.corner(samples_g,
                truths=two_truths,
                labels=two_labels,
                fig=fig
                );
    fig.savefig("../outputs/radio_corner_g.pdf")
    plt.close()

    print("ESS g1", blackjax.diagnostics.effective_sample_size(samples_["g1"][...,0]))
    print("ESS g2", blackjax.diagnostics.effective_sample_size(samples_["g2"][...,0]))
    print("ESS hlr", blackjax.diagnostics.effective_sample_size(samples_["hlr"][...,0]))
    print("ESS flux", blackjax.diagnostics.effective_sample_size(samples_["flux"][...,0]))
    print("ESS e1", blackjax.diagnostics.effective_sample_size(samples_["e1"][...,0]))
    print("ESS e2", blackjax.diagnostics.effective_sample_size(samples_["e2"][...,0]))
    print("ESS at the end of second loop", file=log_file)
    print("ESS g1", blackjax.diagnostics.effective_sample_size(samples_["g1"][...,0]), file=log_file)
    print("ESS g2", blackjax.diagnostics.effective_sample_size(samples_["g2"][...,0]), file=log_file)
    print("ESS hlr", blackjax.diagnostics.effective_sample_size(samples_["hlr"][...,0]), file=log_file)
    print("ESS flux", blackjax.diagnostics.effective_sample_size(samples_["flux"][...,0]), file=log_file)
    print("ESS e1", blackjax.diagnostics.effective_sample_size(samples_["e1"][...,0]), file=log_file)
    print("ESS e2", blackjax.diagnostics.effective_sample_size(samples_["e2"][...,0]), file=log_file)

    flatchain = np.std(samples_["g1"], axis=1) < 1e-4
    print("Flatchains:")
    print(flatchain)
    print("Flatchains:", file=log_file)
    print(flatchain, file=log_file)

    # Save samples
    if args.save_samples:
        np.savez("../outputs/radio_samples.npz", **samples_)

    # Save log file
    log_file.close()

    print("Done.")


if __name__ == "__main__":
    main()