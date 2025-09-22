from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from .data_gen_utils import draw_exp_profile, draw_spergel_profile
from .func_utils import to_unit_disk


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
            nu=nu,
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
