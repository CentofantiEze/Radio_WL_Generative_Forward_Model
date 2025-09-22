from functools import partial

import galsim as gs
import h5py
import jax.numpy as jnp
import jax_galsim as galsim
import numpy as np
import numpyro
import numpyro.distributions as dist

from .func_utils import complex_2_stack, to_unit_disk


def draw_exp_profile(hlr, flux, e1, e2, g1, g2, uv_pos, Npx, pixel_scale):
    gal = galsim.Exponential(half_light_radius=hlr, flux=flux)

    # intrinsic ellipticity
    gal = gal.shear(e1=e1, e2=e2)

    # cosmic shear
    gal = gal.shear(g1=g1, g2=g2)

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
    gal = gal.shear(g1=g1, g2=g2)

    # Convert to Fourier space
    gal_kimage = gal.drawKImage(nx=Npx, ny=Npx, scale=2 * np.pi / (Npx * pixel_scale))

    # Get array
    gal_kimage = gal_kimage.array

    # Sample the visibilities
    vis = gal_kimage[uv_pos]

    return complex_2_stack(vis)


def sample_sersic_params(
    Ngal=None, TRECS_fit_dir=None, ell_scale=None, n=1.0, deepshape_dataset_dir=None
):

    if TRECS_fit_dir is not None:
        params = np.load(TRECS_fit_dir, allow_pickle=True)[()]
        u = jnp.ones((Ngal,))  # sampling galaxies all at once

        hlr_fit = params["beta_fit_hlr"]
        hlr = (
            numpyro.sample(
                "hlr", dist.Beta(hlr_fit["a"], hlr_fit["b"]), sample_shape=(Ngal,)
            )
            * hlr_fit["scale"]
            + hlr_fit["loc"]
        )

        flux_fit = params["beta_fit_flux"]
        flux = (
            numpyro.sample(
                "flux", dist.Beta(flux_fit["a"], flux_fit["b"]), sample_shape=(Ngal,)
            )
            * flux_fit["scale"]
            + flux_fit["loc"]
        )

        e1 = numpyro.sample("e1", dist.Normal(0.0 * u, 1.0 * u)) * ell_scale
        e2 = numpyro.sample("e2", dist.Normal(0.0 * u, 1.0 * u)) * ell_scale

        # clipping undefined e and g values
        e = jnp.stack([e1, e2], 0)
        e = to_unit_disk(e)

        n = jnp.ones((Ngal,)) * n  # Exponential profile, n=1
    elif deepshape_dataset_dir is not None:
        # Load the dataset
        with h5py.File(deepshape_dataset_dir, "r") as f:
            print("Keys in the dataset:", list(f.keys()))
            columns = list(f.keys())
            data = {col: f[col][:] for col in columns}
        # Load parameters
        hlr = data["HLR"][:Ngal]
        flux = data["Flux"][:Ngal]
        e = data["input"][:Ngal]
        n = data["Sersic_index"][:Ngal]
    else:
        raise ValueError("Either TRECS_fit_dir or deepshape_dataset_dir must be provided.")

    return hlr, flux, e, n


def gen_sersic_profile(
    Ngal=None,
    Npx=None,
    pixel_scale=None,
    uv_pos=None,
    noise_uv=None,
    TRECS_fit_dir=None,
    deepshape_dataset_dir=None,
    ell_scale=None,
    g1=None,
    g2=None,
    n=1.0,
):

    hlr, flux, e, n = sample_sersic_params(
        Ngal=Ngal,
        TRECS_fit_dir=TRECS_fit_dir,
        ell_scale=ell_scale,
        n=n,
        deepshape_dataset_dir=deepshape_dataset_dir,
    )

    u = jnp.ones((Ngal,))  # Sheared shear for all galaxies
    g_1 = u * g1
    g_2 = u * g2
    # generate galaxy image
    draw = partial(draw_sersic_profile, uv_pos=uv_pos, Npx=Npx, pixel_scale=pixel_scale)
    im_gal = jnp.array(
        [
            draw(
                n=n[i],
                hlr=hlr[i],
                flux=flux[i],
                e1=e[0][i],
                e2=e[1][i],
                g1=g_1[i],
                g2=g_2[i],
            )
            for i in range(Ngal)
        ]
    )
    data_params = {
        "n": n,
        "hlr": hlr,
        "flux": flux,
        "e1": e[0],
        "e2": e[1],
        "g1": g_1,
        "g2": g_2,
    }

    # add Gaussian noise
    return numpyro.sample("obs", dist.Normal(im_gal, noise_uv)), data_params
