from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import checkpoint

from .data_gen_utils import draw_exp_profile, draw_spergel_profile, draw_NN_profile
from .func_utils import to_unit_disk
from pshear.nn.utils import split # type: ignore


# @partial(jax.jit, static_argnums=(0,1,2,3,4))
def model_fn(
    Ngal=None,
    Npx=None,
    pixel_scale=None,
    uv_pos=None,
    noise_uv=None,
    obs=None,
    ell_sigma=None,
    ell_scale=None,
    g_sigma=None,
    g_scale=None,
    hlr_sigma=None,
    hlr_max=None,
    hlr_min=None,
    flux_sigma=None,
    flux_max=None,
    flux_min=None,
    profile_type="exp",
):

    u = jnp.ones((Ngal,))  # sampling galaxies all at once

    # Spergel profile 
    if profile_type == "spergel":
        nu_min = -0.7  # Safety limit (to avoid nu < -1)
        nu_max = 1.0  # Max limit (to avoid numerical issues at high nu)
        nu_z = numpyro.sample("nu", dist.Normal(0.0 * u, 1.0 * u))
        nu = nu_min + jax.nn.sigmoid(nu_z) * (nu_max - nu_min)

    # hlr
    # hlr = jax.nn.softplus((numpyro.sample("hlr", dist.Normal(0.*u, hlr_sigma*u))/hlr_sigma + hlr_offset)) * hlr_scale + hlr_min
    hlr_z = numpyro.sample("hlr", dist.Normal(0.0 * u, hlr_sigma * u))
    hlr = hlr_min + jax.nn.sigmoid(hlr_z / hlr_sigma) * (hlr_max - hlr_min)

    # flux
    # flux = jax.nn.softplus((numpyro.sample("flux", dist.Normal(0.*u, flux_sigma*u))/flux_sigma + flux_offset)) * flux_scale + flux_min
    flux_z = numpyro.sample("flux", dist.Normal(0.0 * u, flux_sigma * u))
    flux = flux_min + jax.nn.sigmoid(flux_z / flux_sigma) * (flux_max - flux_min)

    # ellipticity
    e1 = (
        numpyro.sample("e1", dist.Normal(0.0 * u, ell_sigma * u))
        / ell_sigma
        * ell_scale
    )
    e2 = (
        numpyro.sample("e2", dist.Normal(0.0 * u, ell_sigma * u))
        / ell_sigma
        * ell_scale
    )

    # assuming constant shear across galaxies
    g1 = (
        numpyro.sample("g1", dist.Normal(jnp.zeros((1,)), g_sigma * jnp.ones((1,))))
        * g_scale
        / g_sigma
    )
    g2 = (
        numpyro.sample("g2", dist.Normal(jnp.zeros((1,)), g_sigma * jnp.ones((1,))))
        * g_scale
        / g_sigma
    )

    # clipping undefined e and g values
    e = jnp.stack([e1, e2], 0)
    e = to_unit_disk(e)

    g = jnp.repeat(jnp.stack([g1, g2], 0), Ngal, -1)
    g = to_unit_disk(g)
    if profile_type == "exp":
        draw = partial(draw_exp_profile, uv_pos=uv_pos, Npx=Npx, pixel_scale=pixel_scale)
        im_gal = jax.vmap(draw)(
            hlr=hlr,
            flux=flux,
            e1=e[0],
            e2=e[1],
            g1=g[0],
            g2=g[1],
        )
    elif profile_type == "sersic":
        raise NotImplementedError("Sersic profile not implemented in JAX-Galsim yet.")
    elif profile_type == "spergel":
        draw = partial(draw_spergel_profile, uv_pos=uv_pos, Npx=Npx, pixel_scale=pixel_scale)
        im_gal = jax.vmap(draw)(
            n=nu,
            hlr=hlr,
            flux=flux,
            e1=e[0],
            e2=e[1],
            g1=g[0],
            g2=g[1],
        )
    else:
        raise ValueError("Profile type not recognized.")
   
    return numpyro.sample("obs", dist.Normal(im_gal, noise_uv), obs=obs)

def model_fn_VAE(
    Ngal=None,
    Npx=None,
    pixel_scale_radio=None,
    pixel_scale_vae=None,
    uv_pos=None,
    noise_uv=None,
    obs=None,
    g_sigma=None,
    g_scale=None,
    flux_sigma=None,
    flux_max=None,
    flux_min=None,
    latent_dim=None,
    latent_mean=None,
    autoencoder=None,
    key=None,
    gsparams=None,
    run_type="sequential",
    batch_size=1
):
    z = numpyro.sample("z", dist.Normal(jnp.zeros((Ngal ,latent_dim, latent_dim)), jnp.ones((Ngal ,latent_dim, latent_dim)))) + latent_mean

    # assuming constant shear across galaxies
    g1 = (
        numpyro.sample("g1", dist.Normal(jnp.zeros((1,)), g_sigma * jnp.ones((1,))))
        * g_scale
        / g_sigma
    )
    g2 = (
        numpyro.sample("g2", dist.Normal(jnp.zeros((1,)), g_sigma * jnp.ones((1,))))
        * g_scale
        / g_sigma
    )
    g = jnp.repeat(jnp.stack([g1, g2], 0), Ngal, -1)
    g = to_unit_disk(g)

    # flux
    flux_z = numpyro.sample("flux", dist.Normal(jnp.zeros((Ngal,)), flux_sigma * jnp.ones((Ngal,))))
    flux = flux_min + jax.nn.sigmoid(flux_z / flux_sigma) * (flux_max - flux_min)

    # A key must be provided for the VAE model.
    if key is None:
        raise ValueError("model_fn_VAE requires a 'key' argument for autoencoder decoding.")

    # Random keys for autoencoder decoding
    subkeys = split(key, Ngal)

    # The TracerIntegerConversionError suggests `subkeys` may be a list of arrays.
    # Stacking them into a single JAX array ensures compatibility with `lax.scan` and `vmap`.
    if isinstance(subkeys, list):
        subkeys = jnp.stack(subkeys)

    # JIT the decode method to accelerate VAE inference
    jitted_decode = jax.jit(autoencoder.decode)

    # Create a partial function to bake in the static arguments for draw_NN_profile.
    # This is the key to avoiding the TypeError with JAX transformations.
    draw = partial(draw_NN_profile, 
                   uv_pos=uv_pos, 
                   Npx=Npx, 
                   pixel_scale_radio=pixel_scale_radio, 
                   pixel_scale_vae=pixel_scale_vae, 
                   jitted_decode=jitted_decode, 
                   gsparams=gsparams)

    if run_type == "sequential":
        def scan_body(carry, sliced_inputs):
            z_i, flux_i, g0_i, g1_i, subkey_i = sliced_inputs
            im_gal_i = checkpoint(draw)(z_i, flux_i, g0_i, g1_i, subkey_i)
            return carry, im_gal_i
        scan_inputs = (z, flux, g[0], g[1], subkeys)
        _, im_gal = jax.lax.scan(scan_body, None, scan_inputs)

    elif run_type == "parallel":
        im_gal = jax.vmap(draw)(z, flux, g[0], g[1], subkeys)

    elif run_type == "batch":
        # Pad inputs to be divisible by batch_size
        pad_size = (batch_size - (Ngal % batch_size)) % batch_size
        if pad_size > 0:
            z = jnp.pad(z, ((0, pad_size), (0, 0), (0, 0)), mode='constant')
            flux = jnp.pad(flux, (0, pad_size), mode='constant')
            g = jnp.pad(g, ((0, 0), (0, pad_size)), mode='constant')
        
        num_batches = z.shape[0] // batch_size
        subkeys = split(key, num_batches)
        
        # Reshape for batching
        z_batched = z.reshape((num_batches, batch_size, latent_dim, latent_dim))
        flux_batched = flux.reshape((num_batches, batch_size))
        g_batched = g.reshape((2, num_batches, batch_size))
        g_scan_inp = jnp.transpose(g_batched, (1, 0, 2))

        vmapped_draw = jax.vmap(draw)

        def batch_scan_body(carry, batch_inputs):
            z_b, flux_b, g_b, subkeys = batch_inputs
            keysbatch = split(subkeys, batch_size)
            im_gal_batch = vmapped_draw(z_b, flux_b, g_b[0], g_b[1], keysbatch)
            return carry, im_gal_batch

        scan_inputs = (z_batched, flux_batched, g_scan_inp, subkeys)
        _, im_gal_batched = jax.lax.scan(batch_scan_body, None, scan_inputs)

        # Reshape and truncate padding
        im_gal_padded = im_gal_batched.reshape((-1, 2,im_gal_batched.shape[-1]))
        im_gal = im_gal_padded[:Ngal]

    else:
        raise ValueError("run_type must be 'sequential', 'parallel', or 'batch'")

    return numpyro.sample("obs", dist.Normal(im_gal, noise_uv), obs=obs)