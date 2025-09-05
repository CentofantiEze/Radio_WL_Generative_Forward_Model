# Radio PSF
import argosim
import argosim.antenna_utils
import argosim.imaging_utils
import numpy as np
# antenna = argosim.antenna_utils.y_antenna_arr(n_antenna=6, r=1e3)
# antenna = argosim.antenna_utils.random_antenna_arr(n_antenna=80, E_lim=50e3, N_lim=50e3)
# antenna = argosim.antenna_utils.uni_antenna_array(n_antenna_E=10, n_antenna_N=4, E_lim=1e3, N_lim=3e3)

def compute_radio_uv_mask(n_antenna=50, 
                          E_lim=50e3, 
                          N_lim=50e3, 
                          Npx=128, 
                          fov_size=0.1,
                          track_time=10,
                          n_times=4,
                          f=1.4e9,
                          df=1e8,
                          n_freqs=4,
                          seed=None):
    antenna = argosim.antenna_utils.random_antenna_arr(n_antenna=n_antenna, E_lim=E_lim, N_lim=N_lim, seed=seed)
    b_enu = argosim.antenna_utils.get_baselines(antenna)
    track, _ = argosim.antenna_utils.uv_track_multiband(b_ENU=b_enu, track_time=track_time, n_times=n_times, f=f, df=df, n_freqs=n_freqs)
    mask, _ = argosim.imaging_utils.grid_uv_samples(track, sky_uv_shape=(Npx, Npx), fov_size=(fov_size, fov_size))
    uv_pos = np.where(np.abs(mask) > 0.)
    psf = np.abs(argosim.imaging_utils.uv2sky(mask))

    return uv_pos, mask, psf
