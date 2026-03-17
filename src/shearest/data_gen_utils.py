from functools import partial

import galsim as gs
import h5py
import jax
import jax.numpy as jnp
import jax_galsim as galsim # type: ignore
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

def draw_spergel_profile(n, hlr, flux, e1, e2, g1, g2, uv_pos, Npx, pixel_scale):
    gal = galsim.Spergel(nu=n, half_light_radius=hlr, flux=flux)

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

def draw_HST_profiles(Ngal, dataset_dir, flux_batch, g1, g2, uv_pos, Npx, pixel_scale_hst=0.03, profile_type="real", sample="23.5", seed=None):

    catalog = gs.COSMOSCatalog(sample=sample, dir=dataset_dir, min_flux=20., min_hlr=0.2, max_hlr=1.)
    rng = np.random.default_rng(seed)
    indices = rng.choice(catalog.getNObjects(), Ngal, replace=False)
    im_gal = []
    if profile_type == "cosmos":
        gal_type = 'parametric'
    else:
        gal_type = 'real'
    for i, ind in enumerate(indices):
        gal_ = catalog.makeGalaxy(ind, gal_type=gal_type)
        # Use the un-convolved galaxy profile
        # psf = gal_.original_psf
        # gal_ = gs.Convolve([gal_, psf])
        # Use the original flux
        # Convolve with gaussian to ensure finite support in Fourier space (for numerical stability)
        gal_ = gs.Convolve([gal_, gs.Gaussian(sigma=2*pixel_scale_hst)])
        gal_ = gal_.withFlux(flux_batch[i])
        gal_ = gal_.shear(g1=g1, g2=g2)
        gal_kimage_ = gal_.drawKImage(nx=Npx, ny=Npx, scale=2*np.pi/pixel_scale_hst/Npx).array
        vis = gal_kimage_[uv_pos]
        im_gal.append(complex_2_stack(vis))
    return jnp.array(im_gal), indices

####### Deprecated functions for direct image manipulation (used galsim-based) #######
# def apply_shear(image, g1, g2, order=1):
#     """Apply area-preserving shear to an image via inverse coordinate mapping.

#     For reduced shear g = (g1, g2), the inverse shear matrix is:
#         S^{-1} = 1/sqrt(1-|g|^2) * [[1-g1, -g2], [-g2, 1+g1]]
#     For each output pixel, we find its source coordinate and interpolate.

#     Args:
#         image: 2D array (Ny, Nx).
#         g1, g2: Reduced shear components.
#         order: Interpolation order (1=bilinear, 3=bicubic).

#     Returns:
#         Sheared image, same shape as input.
#     """
#     ny, nx = image.shape
#     cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0

#     iy, ix = jnp.meshgrid(jnp.arange(ny), jnp.arange(nx), indexing='ij')
#     dy = iy - cy
#     dx = ix - cx

#     g_sq = g1**2 + g2**2
#     inv_sqrt = 1.0 / jnp.sqrt(1.0 - g_sq)
#     dx_src = ((1 - g1) * dx - g2 * dy) * inv_sqrt
#     dy_src = (-g2 * dx + (1 + g1) * dy) * inv_sqrt

#     coords = jnp.stack([dy_src + cy, dx_src + cx])
#     return jax.scipy.ndimage.map_coordinates(image, coords, order=order, mode='constant')

####### Deprecated functions for direct image manipulation (used galsim-based) #######
# def image_to_kimage(image, pixel_scale, Npx_out, pixel_scale_out):
#     """Zero-pad a real-space image and FFT to produce a k-space image.

#     The output k-image has shape (Npx_out, Npx_out) with k-grid spacing
#     dk = 2*pi / (Npx_out * pixel_scale_out).

#     Args:
#         image: 2D array (Ny, Nx) in surface brightness units (flux/area).
#         pixel_scale: Pixel scale of the input image (arcsec/pixel).
#         Npx_out: Side length of the output k-image.
#         pixel_scale_out: Pixel scale defining the output k-grid spacing.

#     Returns:
#         Complex 2D array (Npx_out, Npx_out) — the Fourier transform of the
#         image, following the GalSim convention I_k = integral I(x) exp(-i k.x) dx.
#     """
#     ny, nx = image.shape

#     # Zero-pad so the FFT dk matches the desired output dk.
#     # dk_native = 2*pi / (ny * pixel_scale)
#     # dk_out    = 2*pi / (Npx_out * pixel_scale_out)
#     # N_pad     = Npx_out * pixel_scale_out / pixel_scale
#     N_pad = round(Npx_out * pixel_scale_out / pixel_scale)

#     pad_y = (N_pad - ny) // 2
#     pad_x = (N_pad - nx) // 2
#     y_padded = jnp.pad(image,
#                        ((pad_y, N_pad - ny - pad_y),
#                         (pad_x, N_pad - nx - pad_x)))

#     # FFT with GalSim normalization (continuous FT approximation)
#     y_k = jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(y_padded))) * pixel_scale**2

#     # Extract the central Npx_out x Npx_out region
#     start = (N_pad - Npx_out) // 2
#     return y_k[start:start+Npx_out, start:start+Npx_out]


def draw_NN_profile(z, flux, g1, g2, key, uv_pos, Npx, pixel_scale_vae=0.03, jitted_decode=None, gsparams=None, use_dropout=False):
    # Decode the latent vector to get the galaxy image
    y = jitted_decode(z[None,:,:], key=key if use_dropout else None).astype(jnp.float32)

    # Interpolate Image to galsim object
    y_gs = galsim.InterpolatedImage(
        galsim.Image(y[0], scale=pixel_scale_vae),
        gsparams=gsparams,
        pad_factor=1.0,
        _force_stepk=2 * np.pi / (Npx * pixel_scale_vae),
        _force_maxk=np.pi / pixel_scale_vae
    )

    # Apply shear
    y_gs = y_gs.shear(g1=g1, g2=g2)
    
    # Set flux
    y_gs = y_gs.withFlux(flux)
    
    # Draw kimage
    y_kimage = y_gs.drawKImage(nx=Npx, ny=Npx, scale=2*np.pi/pixel_scale_vae/Npx)

    # Get array
    y_kimage_array = y_kimage.array
    
    # Sample visibilities
    vis = y_kimage_array[uv_pos]

    return complex_2_stack(vis)

def sample_galaxy_params(
    Ngal=None, TRECS_fit_dir=None, ell_scale=None, deepshape_dataset_dir=None, profile_type=None, n=None
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
        if n:
            n = jnp.ones((Ngal,)) * n
        elif profile_type == 'exp':
            n = jnp.ones((Ngal,)) * 1.0  # Exponential profile, n=1.
        elif profile_type == 'spergel':
            nu_min = -0.7  # Safety limit (to avoid nu < -1)
            nu_max = 1.0  # Max limit (to avoid numerical issues at high nu)
            n = nu_min + jax.nn.sigmoid(numpyro.sample("n", dist.Normal(0.0 * u, 1.0 * u))) * (nu_max - nu_min)
        elif profile_type == 'sersic':
            n_min = 0.5  # Safety limit (to avoid nu < -1)
            n_max = 4.0  # Max limit (to avoid numerical issues at high nu)
            n = n_min + jax.nn.sigmoid(numpyro.sample("n", dist.Normal(0.0 * u, 1.0 * u))) * (n_max - n_min)
    elif deepshape_dataset_dir is not None:
        # Load the dataset
        with h5py.File(deepshape_dataset_dir, "r") as f:
            print("Keys in the dataset:", list(f.keys()))
            columns = list(f.keys())
            data = {col: f[col][:] for col in columns}
        # Load parameters
        hlr = data["HLR"][:Ngal]
        flux = data["Flux"][:Ngal] * 1e3  # mJy to uJy
        e = np.array(data["input"][:Ngal]).T  # shape (Ngal, 2) -> (2, Ngal)
        n = data["Sersic_index"][:Ngal]
    else:
        raise ValueError("Either TRECS_fit_dir or deepshape_dataset_dir must be provided.")

    return hlr, flux, e, n

def gen_gal_dataset(
    Ngal=None,
    Npx=None,
    pixel_scale=None,
    uv_pos=None,
    noise_uv=None,
    TRECS_fit_dir=None,
    deepshape_dataset_dir=None,
    cosmos_dataset_dir=None,
    cosmos_sample=None,
    ell_scale=None,
    g1=None,
    g2=None,
    profile_type=None,
    n=None,
    cosmos_seed=None,
):

    hlr_batch, flux_batch, e_batch, n_batch = sample_galaxy_params(
        Ngal=Ngal,
        TRECS_fit_dir=TRECS_fit_dir,
        ell_scale=ell_scale,
        deepshape_dataset_dir=deepshape_dataset_dir,
        profile_type=profile_type,
        n=n,
    )

    u = jnp.ones((Ngal,))  # Sheared shear for all galaxies
    g_1 = u * g1
    g_2 = u * g2
    # generate galaxy image
    if profile_type == "real" or profile_type == "cosmos":
        
        im_gal, indices = draw_HST_profiles(
            Ngal=Ngal,
            dataset_dir=cosmos_dataset_dir,
            flux_batch=flux_batch,
            g1=g1,
            g2=g2,
            uv_pos=uv_pos,
            Npx=Npx,
            profile_type=profile_type,
            sample=cosmos_sample,
            seed=cosmos_seed,
        )
        data_params = {
            "profile_type": profile_type,
            "indices": indices,
            "flux": flux_batch,
            "g1": g1,
            "g2": g2,
        }
    else:
        if profile_type == "exp" or profile_type == "sersic":
            draw = partial(draw_sersic_profile, uv_pos=uv_pos, Npx=Npx, pixel_scale=pixel_scale)
        elif profile_type == "spergel":
            if deepshape_dataset_dir is not None:
                raise ValueError("Spergel profile not available for DeepShape dataset.")
            draw = partial(draw_spergel_profile, uv_pos=uv_pos, Npx=Npx, pixel_scale=pixel_scale)
        else:
            raise ValueError("Profile type not recognized.")
        im_gal = jnp.array(
            [
                draw(
                    n=n_batch[i],
                    hlr=hlr_batch[i],
                    flux=flux_batch[i],
                    e1=e_batch[0][i],
                    e2=e_batch[1][i],
                    g1=g_1[i],
                    g2=g_2[i],
                )
                for i in range(Ngal)
            ]
        )
        data_params = {
            "profile_type": profile_type,
            "n": n_batch,
            "hlr": hlr_batch,
            "flux": flux_batch,
            "e1": e_batch[0],
            "e2": e_batch[1],
            "g1": g_1,
            "g2": g_2,
        }

    # add Gaussian noise
    return numpyro.sample("obs", dist.Normal(im_gal, noise_uv)), data_params
