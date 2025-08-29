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

params_dir = '../data/trecs_gal_params.npy'

def clip_by_l2_norm(x):
    norm = jnp.linalg.norm(x, axis=0)
    scale = jnp.minimum(1.0, 1. / (norm + 1e-2))  # adding epsilon for numerical stability and enforcing norm < 1.
    return x * scale

def complex_2_stack(x):
    """Convert a complex image to a stack of real and imaginary parts."""
    return jnp.stack([jnp.real(x), jnp.imag(x)], axis=0)

def stack_2_complex(x, batch=None):
    """Convert a stack of real and imaginary parts to a complex image."""
    if batch is None:
        return jnp.complex64(x[0] + 1j * x[1])
    else:
        return jnp.complex64(x[:,0] + 1j * x[:,1])

x = jnp.array([0.9558077, -0.8615997])
print(jnp.linalg.norm(x))
print(jnp.linalg.norm(clip_by_l2_norm(x)))

def draw_exp_profile(hlr, flux, e1, e2, g1, g2, uv_pos, Npx, pixel_scale):
    gal = galsim.Exponential(half_light_radius=hlr, flux=flux)
    
    # intrinsic ellipticity
    gal = gal.shear(e1=e1, e2=e2)

    # cosmic shear
    gal = gal.shear(g1=g1,g2=g2)

    # Convert to Fourier space
    gal_kimage = gal.drawKImage(nx=Npx, ny=Npx, scale=2 * jnp.pi / (Npx * pixel_scale))
    
    # Get array
    gal_kimage = gal_kimage.array 

    # Sample the visibilities
    vis = gal_kimage[uv_pos]
    
    return complex_2_stack(vis)

def draw_sersic_profile(hlr, flux, e1, e2, g1, g2, uv_pos, Npx, pixel_scale):
    n = np.random.rand() * 0. + 1. # Exponential profile, n=1
    gal = gs.Sersic(n=n, half_light_radius=hlr, flux=flux)
    
    # intrinsic ellipticity
    gal = gal.shear(e1=e1, e2=e2)

    # cosmic shear
    gal = gal.shear(g1=g1,g2=g2)

    # Convert to Fourier space
    gal_kimage = gal.drawKImage(nx=Npx, ny=Npx, scale=2 * np.pi / (Npx * pixel_scale))
    
    # Get array
    gal_kimage = gal_kimage.array 
    
    # Sample the visibilities
    vis = gal_kimage[uv_pos]

    return complex_2_stack(vis)

ell_prior_scale = 1.
ell_scale = 0.3
g_prior_scale = 1.
g_scale = 0.3
hlr_offset = 2.5
hlr_scale = 1/2.5
hlr_min = 0.1
flux_offset = 1.
flux_scale = 1/30.
flux_min = 0.1
#@partial(jax.jit, static_argnums=(0,1,2,3,4))
def model_fn(Ngal=10, Npx=128, pixel_scale=0.15, uv_pos=None, noise_uv=1e-2, obs=None, params_dir=params_dir, data_gen=False):

    # define parameters and their prior
    params = np.load(params_dir, allow_pickle=True)[()]
    # pixel_scale = params['pixelscale']
    # Npx = params['Npix']

    u = jnp.ones((Ngal,)) # sampling galaxies all at once

    hlr_fit = params['beta_fit_hlr']
    # hlr = numpyro.sample("hlr", dist.Beta(hlr_fit['a'], hlr_fit['b']), sample_shape=(Ngal,)) * hlr_fit['scale'] + hlr_fit['loc']
    # hlr = jnp.abs((numpyro.sample("hlr", dist.Normal(0.*u, 1.*u)) + 4.) * 1.) + 1e-3
    hlr = jnp.abs((numpyro.sample("hlr", dist.Normal(0.*u, 1.*u)) + hlr_offset)) * hlr_scale + hlr_min
    
    flux_fit = params['beta_fit_flux']
    # flux = numpyro.sample("flux", dist.Beta(flux_fit['a'], flux_fit['b']), sample_shape=(Ngal,)) * flux_fit['scale'] + flux_fit['loc']
    # flux = jnp.abs((numpyro.sample("flux", dist.Normal(0.*u, 1.*u)) + 10.) * 2.) + 1e-3
    flux = jnp.abs((numpyro.sample("flux", dist.Normal(0.*u, 1.*u)) + flux_offset)) * flux_scale + flux_min
    
    # r_ell_fit = params['beta_fit_r_ell']
    # r_ell = numpyro.sample("r_ell", dist.Beta(r_ell_fit['a'], r_ell_fit['b']), sample_shape=(Ngal,)) * r_ell_fit['scale'] + r_ell_fit['loc']
    # angle_ell = numpyro.sample("angle_ell", dist.Uniform(-jnp.pi, jnp.pi), sample_shape=(Ngal,))

    # e1 = r_ell * jnp.cos(angle_ell)
    # e2 = r_ell * jnp.sin(angle_ell)
    e1 = numpyro.sample("e1", dist.Normal(0.*u, ell_prior_scale*u))/ell_prior_scale * ell_scale
    e2 = numpyro.sample("e2", dist.Normal(0.*u, ell_prior_scale*u))/ell_prior_scale * ell_scale


    # assuming constant shear across galaxies
    g1 = numpyro.sample("g1", dist.Normal(jnp.zeros((1,)), g_prior_scale*jnp.ones((1,))))*g_scale/g_prior_scale
    g2 = numpyro.sample("g2", dist.Normal(jnp.zeros((1,)), g_prior_scale*jnp.ones((1,))))*g_scale/g_prior_scale

    # clipping undefined e and g values
    e = jnp.stack([e1, e2], 0)
    e = clip_by_l2_norm(e)

    g = jnp.repeat(jnp.stack([g1, g2], 0), Ngal, -1)
    g = clip_by_l2_norm(g)

    # generate galaxy image
    if data_gen:
        draw = partial(draw_sersic_profile, uv_pos=uv_pos, Npx=Npx, pixel_scale=pixel_scale)
        im_gal = jnp.array([draw(hlr=hlr[i], flux=flux[i], e1=e[0][i], e2=e[1][i], g1=g[0][i], g2=g[1][i]) for i in range(Ngal)])
        data_params = {"hlr": hlr, "flux": flux, "e1": e[0], "e2": e[1], "g1": g[0], "g2": g[1]}
        np.save("../outputs/radio_data_params.npy", data_params)
    else:
        draw = partial(draw_exp_profile, uv_pos=uv_pos, Npx=Npx, pixel_scale=pixel_scale)
        im_gal = jax.vmap(draw)(hlr=hlr,
                            flux=flux, 
                            e1=e[0], 
                            e2=e[1],
                            g1=g[0], 
                            g2=g[1],)
    
    # add Gaussian noise
    if obs is None:
        return numpyro.sample("obs", dist.Normal(im_gal, noise_uv))
    else:
        return numpyro.sample("obs", dist.Normal(im_gal, noise_uv), obs=data)


# Dine simulation parameters
Ngal = 100
Npx = 128
pixel_scale = 0.15 # in arcsec/pixel
fov_size = Npx * pixel_scale / 3600 # in degrees
noise_uv = .004
# noise = 1e-5
g1_true = -0.05
g2_true = 0.05

# create log file
log_file = open("../outputs/radio_sampling.log", "w")

# print parameters to log file
print(f"Ngal: {Ngal}", file=log_file)
print(f"Npx: {Npx}", file=log_file)
print(f"pixel_scale: {pixel_scale}", file=log_file)
print(f"fov_size: {fov_size}", file=log_file)
print(f"noise_uv: {noise_uv}", file=log_file)
print(f"g1_true: {g1_true}", file=log_file)
print(f"g2_true: {g2_true}", file=log_file)
print(f"Ellipticity prior scale: {ell_prior_scale}", file=log_file)
print(f"Shear prior scale: {g_prior_scale}", file=log_file)

# Radio PSF
import argosim
import argosim.antenna_utils
import argosim.imaging_utils
# antenna = argosim.antenna_utils.y_antenna_arr(n_antenna=6, r=1e3)
# antenna = argosim.antenna_utils.random_antenna_arr(n_antenna=80, E_lim=50e3, N_lim=50e3)
# antenna = argosim.antenna_utils.uni_antenna_array(n_antenna_E=10, n_antenna_N=4, E_lim=1e3, N_lim=3e3)
antenna = argosim.antenna_utils.random_antenna_arr(n_antenna=50, E_lim=50e3, N_lim=50e3, seed=123)
b_enu = argosim.antenna_utils.get_baselines(antenna)
track, _ = argosim.antenna_utils.uv_track_multiband(b_ENU=b_enu, track_time=10, n_times=4, f=1.4e9, df=1e8, n_freqs=4)
mask, _ = argosim.imaging_utils.grid_uv_samples(track, sky_uv_shape=(Npx, Npx), fov_size=(fov_size, fov_size))
# mask = np.ones_like(mask)
uv_pos = np.where(np.abs(mask) > 0.)

plt.subplots(1,3, figsize=(12, 4))
plt.subplot(131)
plt.imshow(np.real(mask))
plt.title('UV mask')
plt.colorbar()
plt.subplot(132)
plt.imshow(np.abs(argosim.imaging_utils.uv2sky(mask)))
plt.title('Radio PSF')
plt.colorbar()
plt.subplot(133)
plt.imshow(galsim.Gaussian(flux=1., sigma=.2).drawImage(nx=Npx, ny=Npx, scale=pixel_scale).array)
plt.title('Gaussian PSF')
plt.colorbar()
plt.savefig("../outputs/radio_psf.pdf")

# Generate observations
key = jax.random.PRNGKey(42)
model_data_gen = partial(model_fn, Ngal=Ngal, Npx=Npx, pixel_scale=pixel_scale,  uv_pos=uv_pos, noise_uv=noise_uv, data_gen=True) 
seeded_model_data_gen = seed(model_data_gen, key)

# Conditioning model to generate observation with [g1, g2]
conditionned_model = condition(seeded_model_data_gen, {"g1":g1_true*jnp.ones((1,))/(g_scale/g_prior_scale), "g2":g2_true*jnp.ones((1,))/(g_scale/g_prior_scale)})
data = conditionned_model()

# Reset model for sampling
key, subkey = jax.random.split(key)
model = partial(model_fn, Ngal=Ngal, Npx=Npx, pixel_scale=pixel_scale,  uv_pos=uv_pos, noise_uv=noise_uv, data_gen=False)
seeded_model = seed(model, subkey)

# Save the data
np.save("../outputs/radio_data.npy", data)
np.save("../outputs/radio_psf_mask.npy", mask)

# Plot observations
# data_complex = stack_2_complex(data, batch=True)
data_complex = []
for vis in stack_2_complex(data, batch=True):
    img_aux = np.zeros_like(mask)
    img_aux[uv_pos] = vis
    data_complex.append(img_aux)
data_ = rearrange(data_complex, "(n1 n2) h w -> (n1 h) (n2 w)", n1=10, n2=10)
# data_ = rearrange(data_complex, "(n1 n2) h w -> (n1 h) (n2 w)", n1=10, n2=10)
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
idx = np.random.randint(0, Ngal)
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

num_chains = 10

keys = jax.random.split(key, num_chains)[:num_chains]
init_val_ = jax.vmap(draw_params)(keys)
np.save("../outputs/radio_init_val.npy", init_val_, allow_pickle=True)

# Get the log prob of the joint distribution, conditioned on data

@jax.jit
def log_prob_fn(params):
    return numpyro.infer.util.log_density(model, (), {"obs":data,}, params)[0]

# MAP params
lr_map = 2e-3
n_steps_map = 5_000
print(f"MAP learning rate: {lr_map}", file=log_file)
print(f"MAP number of steps: {n_steps_map}", file=log_file)

# find the MAP for chain initialization
nll = lambda params: -log_prob_fn(params)

def find_map(init_params):
    start_learning_rate = lr_map
    optimizer = optax.adafactor(start_learning_rate)

    opt_state = optimizer.init(init_params)

    # A simple update loop.
    def update_step(carry, xs):
        params, opt_state = carry
        grads = jax.grad(nll)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), 0.

    (params, _) , _ = jax.lax.scan(update_step, (init_params, opt_state), length=n_steps_map)

    return params

init_val = jax.vmap(find_map)(init_val_)

print(init_val["g1"]*(g_scale/g_prior_scale), init_val["g2"]*(g_scale/g_prior_scale))
print(f"Initial guess: g1={init_val['g1']*(g_scale/g_prior_scale)}, g2={init_val['g2']*(g_scale/g_prior_scale)}", file=log_file)
np.save("../outputs/radio_map_val.npy", init_val, allow_pickle=True)

# Plot the initial guess for the shear
plt.figure()
plt.scatter(init_val_["g1"]*(g_scale/g_prior_scale), init_val_["g2"]*(g_scale/g_prior_scale))
plt.scatter(init_val["g1"]*(g_scale/g_prior_scale), init_val["g2"]*(g_scale/g_prior_scale))
plt.scatter(g1_true, g2_true, color='red', label='True shear')
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
            num_chains=num_chains,
        )

key_warmup, key_sample = jax.random.split(key)

(last_states, parameters), _ = warmup.run(
            key_warmup,
            init_val,
            num_steps=1000,
        )

print('Step size:',parameters["step_size"])
print(f"Step size: {parameters['step_size']}", file=log_file)
parameters["step_size"] = 0.005
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
num = 20
num_steps = 10_000
print(f"Number of chains: {num_chains}", file=log_file)
print(f"Number of loops: {num}", file=log_file)
print(f"Number of steps: {num_steps}", file=log_file)
print(f"Number of samples per chain: {num_steps*num*2}", file=log_file)

key_chains = jax.random.split(key_sample, num_chains)

last_states, _ = jax.vmap(lambda init_states, keys: run_hmc(init_states, keys, 1))(last_states, key_chains)

sample_list = []

keys = jax.vmap(jax.random.split, in_axes=(0,None))(key_chains,num)

for i in range(num):
    print("Chain", i+1, "of", 2*num, "running...")
    last_states, (samples, info) = jax.vmap(lambda init_states, keys: run_hmc(init_states, keys, num_steps))(last_states, keys[:,i,:])
    sample_list.append(samples)

samples_ = {key: np.concatenate([sample_list[k].position[key] for k in range(num)], 1) for key in last_states.position}
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
for i in range(num):
    print("Extra chain", num+i+1, "of", 2*num, "running...")
    last_states, (samples, info) = jax.vmap(lambda init_states, keys: run_hmc(init_states, keys, num_steps))(last_states, keys[:,i,:])
    sample_list.append(samples)

# concatenates chains
samples_ = {key: np.concatenate([sample_list[k].position[key] for k in range(num*2)], 1) for key in last_states.position}

# labels = ["hlr", "flux", "r_ell", "angle_ell", "g1", "g2"]
labels = ["hlr", "flux", "e1", "e2", "g1", "g2"]
params_scales = np.load(params_dir, allow_pickle=True)[()]

fig, axes = plt.subplots(len(labels), figsize=(10, 7), sharex=True)

for i, label in enumerate(labels):
    print(i, label)
    ax = axes[i]
    for k in range(num_chains):
        # if label in ["g1", "g2"]:
        #     ax.plot(samples_[label][k,:,0]*0.1, "k", alpha=0.3)
        # else:
        #     ax.plot(samples_[label][k,:,0], "k", alpha=0.3)
        ax.plot(samples_[label][k,:,0], "k", alpha=0.3)
    ax.set_xlim(0, num_steps*num*2)
    ax.set_ylabel(label)
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")
plt.savefig("../outputs/radio_chains.pdf")

two_truths = np.array([g1_true, g2_true])
samples_g = np.concatenate([samples_["g1"], samples_["g2"]], -1).reshape((-1,2)) * (g_scale/g_prior_scale)

import corner

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
# np.savez("../outputs/radio_samples.npz", **samples_)

# Save log file
log_file.close()
