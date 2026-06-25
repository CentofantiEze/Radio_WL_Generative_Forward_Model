from functools import partial

import galsim as gs
from galsim.hsm import FindAdaptiveMom
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

def draw_composite_profile(hlr_disk, hlr_bulge, flux, flux_db_ratio, e_bulge_1, e_bulge_2, e_disk_1, e_disk_2, g1, g2, uv_pos, Npx, pixel_scale):
    # Draw bulge component with flux set to 1.0 (we will normalize the total flux later)
    gal_bulge = galsim.Spergel(nu=-0.6, half_light_radius=hlr_bulge, flux=1.0)

    # Draw disk component with flux set to the desired bulge-to-disk ratio (we will normalize the total flux later)
    gal_disk = galsim.Spergel(nu=0.5, half_light_radius=hlr_disk, flux=flux_db_ratio)

    # Apply intrinsic ellipticities
    gal_disk = gal_disk.shear(e1=e_disk_1, e2=e_disk_2)
    gal_bulge = gal_bulge.shear(e1=e_bulge_1, e2=e_bulge_2)

    # Combine components
    gal = gal_disk + gal_bulge

    # Set total flux
    gal = gal.withFlux(flux)

    # Apply cosmic shear
    gal = gal.shear(g1=g1, g2=g2)

    # Convert to Fourier space
    gal_kimage = gal.drawKImage(nx=Npx, ny=Npx, scale=2 * np.pi / (Npx * pixel_scale))

    # Get array
    gal_kimage = gal_kimage.array

    # Sample the visibilities
    vis = gal_kimage[uv_pos]

    return complex_2_stack(vis)

def draw_HST_profiles(Ngal, dataset_dir, flux_batch, g1, g2, uv_pos, Npx, pixel_scale_hst=0.03, profile_type="real", sample="25.2", seed=None, mag_cut=None):

    catalog = gs.COSMOSCatalog(sample=sample, dir=dataset_dir)
    rng = np.random.default_rng(seed)
    if mag_cut is not None:
        mag_cut_list = np.where(catalog.param_cat[catalog.orig_index]['mag_auto'] > mag_cut)[0]
        print(f'{len(mag_cut_list)} sources with mag_auto > {mag_cut}.')
        indices = rng.choice(mag_cut_list, Ngal, replace=False)
    else:
        indices = rng.choice(catalog.getNObjects(), Ngal, replace=False)
    im_gal = []
    if profile_type == "cosmos":
        gal_type = 'parametric'
    else:
        gal_type = 'real'
    for i, ind in enumerate(indices):
        gal_ = catalog.makeGalaxy(ind, gal_type=gal_type)
        # DEBUG ONLY: recenter via HSM adaptive moments on the PSF-convolved
        # image. gal_.centroid is noise-dominated for some COSMOS galaxies
        # (deconvolution by RealGalaxy amplifies noise far from center and
        # pulls the intensity-weighted moment off by arcsec); HSM uses
        # Gaussian-weighted moments and stays sub-pixel.
        try:
            psf_ = gal_.original_psf
            im_for_mom = gs.Convolve(gal_, psf_).drawImage(
                nx=Npx, ny=Npx, scale=pixel_scale_hst, method='no_pixel'
            )
            moms = FindAdaptiveMom(im_for_mom)
            center_galsim = (Npx + 1) / 2  # galsim 1-indexed true center
            dx = (moms.moments_centroid.x - center_galsim) * pixel_scale_hst
            dy = (moms.moments_centroid.y - center_galsim) * pixel_scale_hst
            gal_ = gal_.shift(-dx, -dy)
        except gs.errors.GalSimHSMError:
            pass  # HSM failed to converge; leave galaxy uncentered
        gal_ = gal_.shear(g1=g1, g2=g2)
        # Convolve with gaussian to ensure finite support in Fourier space (for numerical stability)
        gal_ = gs.Convolve([gal_, gs.Gaussian(sigma=2*pixel_scale_hst)])
        
        gal_kimage_ = gal_.drawKImage(nx=Npx, ny=Npx, scale=2*np.pi/pixel_scale_hst/Npx).array
        
        # Rescale images by sqrt(# pixels) = Npx
        gal_kimage_ = gal_kimage_ / Npx
        vis = gal_kimage_[uv_pos]
        im_gal.append(complex_2_stack(vis))
    return jnp.array(im_gal), indices

def draw_AE_HST_profiles(Ngal, dataset_dir, ae, g1, g2, uv_pos, Npx,
                          pixel_scale_hst=0.03, pixel_scale_vae=0.03,
                          sample="25.2", seed=None, mag_cut=None):
    """Generate visibilities from COSMOS galaxies passed through the AE.

    For each selected COSMOS galaxy: draw the HST obs+psf stamps (matching the
    training-stamp generation in scripts/examples/cosmos.py), encode through
    the AE, take the decoded intrinsic galaxy `g`, then apply shear + Gaussian
    PSF + drawKImage with regular galsim (matching draw_HST_profiles).

    The result lies on the AE's manifold by construction, so a model that uses
    the same AE for inference can match the data exactly when z = z_enc.

    Returns (visibilities, indices, z_enc).
    """
    catalog = gs.COSMOSCatalog(sample=sample, dir=dataset_dir)
    rng = np.random.default_rng(seed)
    if mag_cut is not None:
        mag_cut_list = np.where(catalog.param_cat[catalog.orig_index]['mag_auto'] > mag_cut)[0]
        print(f'{len(mag_cut_list)} sources with mag_auto > {mag_cut}.')
        indices = rng.choice(mag_cut_list, Ngal, replace=False)
    else:
        indices = rng.choice(catalog.getNObjects(), Ngal, replace=False)

    # Build obs+psf stamps the same way as cosmos.py (AE training data)
    obs_list, psf_list = [], []
    for ind in indices:
        gal_real = catalog.makeGalaxy(int(ind), gal_type='real',
                                       noise_pad_size=Npx * pixel_scale_hst)
        psf = gal_real.original_psf
        psf_im = psf.drawImage(nx=Npx, ny=Npx, scale=pixel_scale_hst,
                               method='no_pixel').array.astype('float32')
        obs_im = gs.Convolve(gal_real, psf).drawImage(
            nx=Npx, ny=Npx, scale=pixel_scale_hst, method='no_pixel'
        ).array.astype('float32')
        obs_list.append(obs_im[None])  # (1, Npx, Npx)
        psf_list.append(psf_im[None])

    obs_arr = jnp.array(np.stack(obs_list))
    psf_arr = jnp.array(np.stack(psf_list))

    # Encode → decode through AE. The encoder returns (y, g, z); we keep g
    # (intrinsic galaxy) for rendering and z (latent) for diagnostics.
    _, g_dec, z_enc = jax.vmap(ae)(obs_arr, psf_arr)
    g_dec_np = np.array(g_dec)  # (Ngal, 1, Npx, Npx)

    # Render via JAX-galsim, matching draw_HST_profiles
    im_gal = []
    for i in range(Ngal):
        gal_ = galsim.InterpolatedImage(galsim.Image(g_dec_np[i, 0], scale=pixel_scale_vae))
        gal_ = gal_.shear(g1=g1, g2=g2)
        gal_ = galsim.Convolve([gal_, galsim.Gaussian(sigma=2 * pixel_scale_vae)])
        gal_kimage_ = gal_.drawKImage(
            nx=Npx, ny=Npx, scale=2 * np.pi / pixel_scale_vae / Npx
        ).array
        # Match the rescale done in draw_HST_profiles / draw_NN_profile
        gal_kimage_ = gal_kimage_ / Npx
        vis = gal_kimage_[uv_pos]
        im_gal.append(complex_2_stack(vis))

    return jnp.array(im_gal), indices, np.array(z_enc)


def draw_NN_profile(z, flux, g1, g2, key, uv_pos, Npx, pixel_scale_vae=0.03, jitted_decode=None, gsparams=None, use_dropout=False):
    # Decode the latent vector to get the galaxy image
    y = jitted_decode(z[None,:,:], key=key if use_dropout else None).astype(jnp.float32)

    # Interpolate Image to galsim object
    # y_gs = galsim.InterpolatedImage(
    #     galsim.Image(y[0], scale=pixel_scale_vae),
    #     gsparams=gsparams,
    #     pad_factor=1.0,
    #     _force_stepk=2 * np.pi / (Npx * pixel_scale_vae),
    #     _force_maxk=np.pi / pixel_scale_vae
    # )
    y_gs = galsim.InterpolatedImage(
        galsim.Image(y[0], scale=pixel_scale_vae),
        gsparams=gsparams,
    )

    # Apply shear
    y_gs = y_gs.shear(g1=g1, g2=g2)

    # Apply PSF convolution
    y_gs = galsim.Convolve([y_gs, galsim.Gaussian(sigma=2*pixel_scale_vae)])
    
    # Draw kimage
    y_kimage = y_gs.drawKImage(nx=Npx, ny=Npx, scale=2*np.pi/pixel_scale_vae/Npx)

    # Get array
    y_kimage_array = y_kimage.array

    # Rescale images by sqrt(# pixels) = Npx
    y_kimage_array = y_kimage_array / Npx
    
    # Sample visibilities
    vis = y_kimage_array[uv_pos]

    return complex_2_stack(vis)

def sample_galaxy_params(
    Ngal=None, TRECS_fit_dir=None, ell_scale=None, deepshape_dataset_dir=None, profile_type=None, n=None
):

    if TRECS_fit_dir is not None:
        params = np.load(TRECS_fit_dir, allow_pickle=True)[()]
        u = jnp.ones((Ngal,))  # sampling galaxies all at once

        # The fit was done with scipy lognorm, which has a different parameterization than Numpyro.
        # log(Y - loc) ~ Normal(log(scale), shape^2)
        hlr_fit = params["lognorm_fit_hlr"]
        lognorm_loc = np.log(hlr_fit["scale"])
        lognorm_scale = hlr_fit["shape"]
        lognorm_shift = hlr_fit["loc"]
        hlr = (
            numpyro.sample(
                "hlr", dist.LogNormal(lognorm_loc, lognorm_scale), sample_shape=(Ngal,)
            )
            + lognorm_shift
        )

        flux_fit = params["expon_fit_flux"]
        flux_rate = 1/flux_fit["scale"]  # Exponential distribution in scipy is parameterized by scale, while in numpyro it's parameterized by rate (1/scale)
        flux_shift = flux_fit["loc"]
        flux = (
            numpyro.sample(
                "flux", dist.Exponential(flux_rate), sample_shape=(Ngal,)
            )
            + flux_shift
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
    pixel_scale_vae=0.03,
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
    mag_cut=None,
    ae=None,
):

    # VAE path skips sample_galaxy_params (no TRECS/deepshape needed) since the
    # galaxy comes from the AE-decoded COSMOS stamp.
    if profile_type == "VAE":
        if ae is None:
            raise ValueError("ae must be provided when profile_type='VAE'")
        if cosmos_dataset_dir is None:
            raise ValueError("cosmos_dataset_dir must be provided when profile_type='VAE'")
        im_gal, indices, z_enc = draw_AE_HST_profiles(
            Ngal=Ngal,
            dataset_dir=cosmos_dataset_dir,
            ae=ae,
            g1=g1,
            g2=g2,
            uv_pos=uv_pos,
            Npx=Npx,
            pixel_scale_vae=pixel_scale_vae,
            sample=cosmos_sample,
            seed=cosmos_seed,
            mag_cut=mag_cut,
        )
        data_params = {
            "profile_type": profile_type,
            "indices": indices,
            "z_enc": z_enc,
            "g1": g1,
            "g2": g2,
        }
        return numpyro.sample("obs", dist.Normal(im_gal, noise_uv)), data_params

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
            flux_batch=None,
            g1=g1,
            g2=g2,
            uv_pos=uv_pos,
            Npx=Npx,
            profile_type=profile_type,
            sample=cosmos_sample,
            seed=cosmos_seed,
            mag_cut=mag_cut,
        )
        data_params = {
            "profile_type": profile_type,
            "indices": indices,
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
