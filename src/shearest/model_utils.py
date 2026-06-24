from functools import partial
from xml.parsers.expat import model

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import checkpoint
import equinox as eqx

from .data_gen_utils import draw_exp_profile, draw_spergel_profile, draw_NN_profile, draw_composite_profile
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

def model_fn_composite(
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
    flux_ratio_max=4.0,
):

    u = jnp.ones((Ngal,))  # sampling galaxies all at once

    # hlr_disk
    hlr_disk_z = numpyro.sample("hlr_disk", dist.Normal(0.0 * u, hlr_sigma * u))
    hlr_disk = hlr_min + jax.nn.sigmoid(hlr_disk_z / hlr_sigma) * (hlr_max - hlr_min)

    # hlr_bulge
    hlr_bulge_z = numpyro.sample("hlr_bulge", dist.Normal(0.0 * u, hlr_sigma * u))
    hlr_bulge = hlr_min + jax.nn.sigmoid(hlr_bulge_z / hlr_sigma) * (hlr_max - hlr_min)

    # flux
    flux_z = numpyro.sample("flux", dist.Normal(0.0 * u, flux_sigma * u))
    flux = flux_min + jax.nn.sigmoid(flux_z / flux_sigma) * (flux_max - flux_min)

    # flux ratio (disk/bulge), reparameterized with sigmoid to [0, 4]
    flux_ratio_z = numpyro.sample("flux_ratio", dist.Normal(0.0 * u, 1.0 * u))
    flux_ratio = jax.nn.sigmoid(flux_ratio_z) * flux_ratio_max

    # ellipticity
    e1_disk = (
        numpyro.sample("e1_disk", dist.Normal(0.0 * u, ell_sigma * u))
        / ell_sigma
        * ell_scale
    )
    e2_disk = (
        numpyro.sample("e2_disk", dist.Normal(0.0 * u, ell_sigma * u))
        / ell_sigma
        * ell_scale
    )
    e1_bulge = (
        numpyro.sample("e1_bulge", dist.Normal(0.0 * u, ell_sigma * u))
        / ell_sigma
        * ell_scale
    )
    e2_bulge = (
        numpyro.sample("e2_bulge", dist.Normal(0.0 * u, ell_sigma * u))
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
    e_bulge = jnp.stack([e1_bulge, e2_bulge], 0)
    e_disk = jnp.stack([e1_disk, e2_disk], 0)
    e_bulge = to_unit_disk(e_bulge)
    e_disk = to_unit_disk(e_disk)

    g = jnp.repeat(jnp.stack([g1, g2], 0), Ngal, -1)
    g = to_unit_disk(g)
    
    
    draw = partial(draw_composite_profile, uv_pos=uv_pos, Npx=Npx, pixel_scale=pixel_scale)
    im_gal = jax.vmap(draw)(
        hlr_disk=hlr_disk,
        hlr_bulge=hlr_bulge,
        flux=flux,
        flux_db_ratio=flux_ratio,
        e_bulge_1=e_bulge[0],
        e_bulge_2=e_bulge[1],
        e_disk_1=e_disk[0],
        e_disk_2=e_disk[1],
        g1=g[0],
        g2=g[1],
    )
   
    return numpyro.sample("obs", dist.Normal(im_gal, noise_uv), obs=obs)

def model_fn_VAE(
    Ngal=None,
    Npx=None,
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
    latent_sigma=1.0,
    jitted_decode=None,
    gsparams=None,
    run_type="sequential",
    batch_size=1,
    use_dropout=False,
):
    z = numpyro.sample("z", dist.Normal(jnp.zeros((Ngal ,latent_dim, latent_dim)), latent_sigma * jnp.ones((Ngal ,latent_dim, latent_dim)))) / latent_sigma + latent_mean

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

    if use_dropout:
        key = numpyro.prng_key()
        subkeys = split(key, Ngal)
    else:
        # Dummy keys as placeholder array for scan/vmap; draw_NN_profile will pass None to decoder
        subkeys = jnp.zeros((Ngal, 2), dtype=jnp.uint32)

    # @eqx.filter_jit
    # def run_decode(model, z, key):
    #     return model.decode(z, key=key)

    # Create a partial function to bake in the static arguments for draw_NN_profile.
    # This is the key to avoiding the TypeError with JAX transformations.
    draw = partial(draw_NN_profile,
                   uv_pos=uv_pos,
                   Npx=Npx,
                   pixel_scale_vae=pixel_scale_vae,
                   jitted_decode=jitted_decode,
                   gsparams=gsparams,
                   use_dropout=use_dropout)

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
        # Pad subkeys to match padded galaxy count, then reshape for batching
        if pad_size > 0:
            subkeys = jnp.pad(subkeys, ((0, pad_size), (0, 0)), mode='constant')
        subkeys_batched = subkeys.reshape((num_batches, batch_size, -1))

        # Reshape for batching
        z_batched = z.reshape((num_batches, batch_size, latent_dim, latent_dim))
        flux_batched = flux.reshape((num_batches, batch_size))
        g_batched = g.reshape((2, num_batches, batch_size))
        g_scan_inp = jnp.transpose(g_batched, (1, 0, 2))

        vmapped_draw = jax.vmap(draw)

        def batch_scan_body(carry, batch_inputs):
            z_b, flux_b, g_b, keys_b = batch_inputs
            im_gal_batch = vmapped_draw(z_b, flux_b, g_b[0], g_b[1], keys_b)
            return carry, im_gal_batch

        scan_inputs = (z_batched, flux_batched, g_scan_inp, subkeys_batched)
        _, im_gal_batched = jax.lax.scan(batch_scan_body, None, scan_inputs)

        # Reshape and truncate padding
        im_gal_padded = im_gal_batched.reshape((-1, 2,im_gal_batched.shape[-1]))
        im_gal = im_gal_padded[:Ngal]

    else:
        raise ValueError("run_type must be 'sequential', 'parallel', or 'batch'")

    return numpyro.sample("obs", dist.Normal(im_gal, noise_uv), obs=obs)


def model_fn_VAE_noshear(
    Ngal=None,
    Npx=None,
    pixel_scale_vae=None,
    uv_pos=None,
    noise_uv=None,
    obs=None,
    flux_sigma=None,
    flux_max=None,
    flux_min=None,
    latent_dim=None,
    latent_mean=None,
    latent_sigma=1.0,
    jitted_decode=None,
    gsparams=None,
    run_type="sequential",
    batch_size=1,
    use_dropout=False,
):
    z = numpyro.sample("z", dist.Normal(jnp.zeros((Ngal, latent_dim, latent_dim)), latent_sigma * jnp.ones((Ngal, latent_dim, latent_dim)))) / latent_sigma + latent_mean

    # No shear — g1=g2=0
    g = jnp.zeros((2, Ngal))

    # flux
    flux_z = numpyro.sample("flux", dist.Normal(jnp.zeros((Ngal,)), flux_sigma * jnp.ones((Ngal,))))
    flux = flux_min + jax.nn.sigmoid(flux_z / flux_sigma) * (flux_max - flux_min)

    if use_dropout:
        key = numpyro.prng_key()
        subkeys = split(key, Ngal)
    else:
        subkeys = jnp.zeros((Ngal, 2), dtype=jnp.uint32)

    draw = partial(draw_NN_profile,
                   uv_pos=uv_pos,
                   Npx=Npx,
                   pixel_scale_vae=pixel_scale_vae,
                   jitted_decode=jitted_decode,
                   gsparams=gsparams,
                   use_dropout=use_dropout)

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
        pad_size = (batch_size - (Ngal % batch_size)) % batch_size
        if pad_size > 0:
            z = jnp.pad(z, ((0, pad_size), (0, 0), (0, 0)), mode='constant')
            flux = jnp.pad(flux, (0, pad_size), mode='constant')
            g = jnp.pad(g, ((0, 0), (0, pad_size)), mode='constant')

        num_batches = z.shape[0] // batch_size
        if pad_size > 0:
            subkeys = jnp.pad(subkeys, ((0, pad_size), (0, 0)), mode='constant')
        subkeys_batched = subkeys.reshape((num_batches, batch_size, -1))

        z_batched = z.reshape((num_batches, batch_size, latent_dim, latent_dim))
        flux_batched = flux.reshape((num_batches, batch_size))
        g_batched = g.reshape((2, num_batches, batch_size))
        g_scan_inp = jnp.transpose(g_batched, (1, 0, 2))

        vmapped_draw = jax.vmap(draw)

        def batch_scan_body(carry, batch_inputs):
            z_b, flux_b, g_b, keys_b = batch_inputs
            im_gal_batch = vmapped_draw(z_b, flux_b, g_b[0], g_b[1], keys_b)
            return carry, im_gal_batch

        scan_inputs = (z_batched, flux_batched, g_scan_inp, subkeys_batched)
        _, im_gal_batched = jax.lax.scan(batch_scan_body, None, scan_inputs)

        im_gal_padded = im_gal_batched.reshape((-1, 2, im_gal_batched.shape[-1]))
        im_gal = im_gal_padded[:Ngal]

    else:
        raise ValueError("run_type must be 'sequential', 'parallel', or 'batch'")

    return numpyro.sample("obs", dist.Normal(im_gal, noise_uv), obs=obs)


# DEBUG ONLY
def model_fn_VAE_flow_noshear(
    Ngal=None,
    Npx=None,
    pixel_scale_vae=None,
    uv_pos=None,
    noise_uv=None,
    obs=None,
    flux_sigma=None,
    flux_max=None,
    flux_min=None,
    latent_dim=None,
    latent_sigma=1.0,
    jitted_decode=None,
    gsparams=None,
    run_type="sequential",
    batch_size=1,
    use_dropout=False,
    flow_forward=None,
    flow_condition=None,
):
    # Sample u in the flow base space (standard normal)
    u = numpyro.sample("u", dist.Normal(jnp.zeros((Ngal, latent_dim, latent_dim)), latent_sigma * jnp.ones((Ngal, latent_dim, latent_dim)))) / latent_sigma

    # Transform u -> z via the flow bijection
    u_flat = u.reshape(Ngal, -1)
    if flow_condition is not None:
        cond_rep = jnp.tile(flow_condition, (Ngal, 1))
        z_flat = jax.vmap(flow_forward)(u_flat, cond_rep)
    else:
        z_flat = jax.vmap(flow_forward)(u_flat)
    z = z_flat.reshape(Ngal, latent_dim, latent_dim)

    # No shear — g1=g2=0
    g = jnp.zeros((2, Ngal))

    # Flux is not used for the VAE
    flux = None

    if use_dropout:
        key = numpyro.prng_key()
        subkeys = split(key, Ngal)
    else:
        subkeys = jnp.zeros((Ngal, 2), dtype=jnp.uint32)

    draw = partial(draw_NN_profile,
                   uv_pos=uv_pos,
                   Npx=Npx,
                   pixel_scale_vae=pixel_scale_vae,
                   jitted_decode=jitted_decode,
                   gsparams=gsparams,
                   use_dropout=use_dropout)

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
        pad_size = (batch_size - (Ngal % batch_size)) % batch_size
        if pad_size > 0:
            z = jnp.pad(z, ((0, pad_size), (0, 0), (0, 0)), mode='constant')
            flux = jnp.pad(flux, (0, pad_size), mode='constant')
            g = jnp.pad(g, ((0, 0), (0, pad_size)), mode='constant')

        num_batches = z.shape[0] // batch_size
        if pad_size > 0:
            subkeys = jnp.pad(subkeys, ((0, pad_size), (0, 0)), mode='constant')
        subkeys_batched = subkeys.reshape((num_batches, batch_size, -1))

        z_batched = z.reshape((num_batches, batch_size, latent_dim, latent_dim))
        flux_batched = flux.reshape((num_batches, batch_size))
        g_batched = g.reshape((2, num_batches, batch_size))
        g_scan_inp = jnp.transpose(g_batched, (1, 0, 2))

        vmapped_draw = jax.vmap(draw)

        def batch_scan_body(carry, batch_inputs):
            z_b, flux_b, g_b, keys_b = batch_inputs
            im_gal_batch = vmapped_draw(z_b, flux_b, g_b[0], g_b[1], keys_b)
            return carry, im_gal_batch

        scan_inputs = (z_batched, flux_batched, g_scan_inp, subkeys_batched)
        _, im_gal_batched = jax.lax.scan(batch_scan_body, None, scan_inputs)

        im_gal_padded = im_gal_batched.reshape((-1, 2, im_gal_batched.shape[-1]))
        im_gal = im_gal_padded[:Ngal]

    else:
        raise ValueError("run_type must be 'sequential', 'parallel', or 'batch'")

    return numpyro.sample("obs", dist.Normal(im_gal, noise_uv), obs=obs)


def model_fn_VAE_flow(
    Ngal=None,
    Npx=None,
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
    latent_sigma=1.0,
    jitted_decode=None,
    gsparams=None,
    run_type="sequential",
    batch_size=1,
    use_dropout=False,
    flow_forward=None,
    flow_condition=None,
):
    # Sample u in the flow base space (standard normal)
    u = numpyro.sample("u", dist.Normal(jnp.zeros((Ngal, latent_dim, latent_dim)), latent_sigma * jnp.ones((Ngal, latent_dim, latent_dim)))) / latent_sigma

    # Transform u -> z via the flow bijection
    u_flat = u.reshape(Ngal, -1)  # (Ngal, latent_dim^2)
    if flow_condition is not None:
        cond_rep = jnp.tile(flow_condition, (Ngal, 1))  # (Ngal, cond_dim)
        z_flat = jax.vmap(flow_forward)(u_flat, cond_rep)
    else:
        z_flat = jax.vmap(flow_forward)(u_flat)
    z = z_flat.reshape(Ngal, latent_dim, latent_dim)

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
    #

    # Flux is not used for the VAE
    flux = None

    if use_dropout:
        key = numpyro.prng_key()
        subkeys = split(key, Ngal)
    else:
        subkeys = jnp.zeros((Ngal, 2), dtype=jnp.uint32)

    draw = partial(draw_NN_profile,
                   uv_pos=uv_pos,
                   Npx=Npx,
                   pixel_scale_vae=pixel_scale_vae,
                   jitted_decode=jitted_decode,
                   gsparams=gsparams,
                   use_dropout=use_dropout)

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
        pad_size = (batch_size - (Ngal % batch_size)) % batch_size
        if pad_size > 0:
            z = jnp.pad(z, ((0, pad_size), (0, 0), (0, 0)), mode='constant')
            flux = jnp.pad(flux, (0, pad_size), mode='constant')
            g = jnp.pad(g, ((0, 0), (0, pad_size)), mode='constant')

        num_batches = z.shape[0] // batch_size
        if pad_size > 0:
            subkeys = jnp.pad(subkeys, ((0, pad_size), (0, 0)), mode='constant')
        subkeys_batched = subkeys.reshape((num_batches, batch_size, -1))

        z_batched = z.reshape((num_batches, batch_size, latent_dim, latent_dim))
        flux_batched = flux.reshape((num_batches, batch_size))
        g_batched = g.reshape((2, num_batches, batch_size))
        g_scan_inp = jnp.transpose(g_batched, (1, 0, 2))

        vmapped_draw = jax.vmap(draw)

        def batch_scan_body(carry, batch_inputs):
            z_b, flux_b, g_b, keys_b = batch_inputs
            im_gal_batch = vmapped_draw(z_b, flux_b, g_b[0], g_b[1], keys_b)
            return carry, im_gal_batch

        scan_inputs = (z_batched, flux_batched, g_scan_inp, subkeys_batched)
        _, im_gal_batched = jax.lax.scan(batch_scan_body, None, scan_inputs)

        im_gal_padded = im_gal_batched.reshape((-1, 2, im_gal_batched.shape[-1]))
        im_gal = im_gal_padded[:Ngal]

    else:
        raise ValueError("run_type must be 'sequential', 'parallel', or 'batch'")

    return numpyro.sample("obs", dist.Normal(im_gal, noise_uv), obs=obs)