import numpy as np
import jax.numpy as jnp
import jax_galsim as galsim
import galsim as gs

from functools import partial
import numpyro
import numpyro.distributions as dist

from func_utils import complex_2_stack
from func_utils import to_unit_disk

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

def draw_sersic_profile(n, hlr, flux, e1, e2, g1, g2, uv_pos, Npx, pixel_scale):
    # n = np.random.rand() * 0. + 4. # Exponential profile, n=1
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

def gen_sersic_profile(Ngal=None, Npx=None, pixel_scale=None, uv_pos=None, noise_uv=None, params_dir=None,
            ell_sigma=None,
            ell_scale=None,
            g_sigma=None,
            g_scale=None,
            n=1.):

    # define parameters and their prior
    params = np.load(params_dir, allow_pickle=True)[()]

    u = jnp.ones((Ngal,)) # sampling galaxies all at once

    hlr_fit = params['beta_fit_hlr']
    hlr = numpyro.sample("hlr", dist.Beta(hlr_fit['a'], hlr_fit['b']), sample_shape=(Ngal,)) * hlr_fit['scale'] + hlr_fit['loc']

    
    flux_fit = params['beta_fit_flux']
    flux = numpyro.sample("flux", dist.Beta(flux_fit['a'], flux_fit['b']), sample_shape=(Ngal,)) * flux_fit['scale'] + flux_fit['loc']
 
    e1 = numpyro.sample("e1", dist.Normal(0.*u, ell_sigma*u))/ell_sigma * ell_scale
    e2 = numpyro.sample("e2", dist.Normal(0.*u, ell_sigma*u))/ell_sigma * ell_scale

    # assuming constant shear across galaxies
    g1 = numpyro.sample("g1", dist.Normal(jnp.zeros((1,)), g_sigma*jnp.ones((1,))))*g_scale/g_sigma
    g2 = numpyro.sample("g2", dist.Normal(jnp.zeros((1,)), g_sigma*jnp.ones((1,))))*g_scale/g_sigma

    # clipping undefined e and g values
    e = jnp.stack([e1, e2], 0)
    e = to_unit_disk(e)

    g = jnp.repeat(jnp.stack([g1, g2], 0), Ngal, -1)
    g = to_unit_disk(g)

    # generate galaxy image
    draw = partial(draw_sersic_profile, n=n, uv_pos=uv_pos, Npx=Npx, pixel_scale=pixel_scale)
    im_gal = jnp.array([draw(hlr=hlr[i], flux=flux[i], e1=e[0][i], e2=e[1][i], g1=g[0][i], g2=g[1][i]) for i in range(Ngal)])
    data_params = {"n": n, "hlr": hlr, "flux": flux, "e1": e[0], "e2": e[1], "g1": g[0], "g2": g[1]}

    # add Gaussian noise
    return numpyro.sample("obs", dist.Normal(im_gal, noise_uv)), data_params

