#!/bin/bash
#SBATCH --mail-user=ezequiel.centofanti@cea.fr
#SBATCH --mail-type=NONE
#SBATCH --job-name=g0-0
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH -C v100-32g
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=out_0-0_%a.out
#SBATCH --error=err_0-0_%a.err
#SBATCH -A prk@v100
#SBATCH --qos=qos_gpu-dev
#SBATCH --array=80-89

module purge
module load anaconda-py3/2024.06
conda activate forwardmodel

set -x

cd ${WORK}/repos/Radio_WL_Generative_Forward_Model/scripts

args=(
    --Ngal 100
    --Npx 128
    --pixel_scale 0.15
    --noise_uv 0.02
    --trecs_data_path ../data/trecs_gal_params.npy
    --data_profile spergel
    --g1_true 0.0
    --g2_true 0.0
    --ell_scale 0.2
    --antenna_type file
    --antenna_file ../data/SKA-Mid.txt
    #--n_antenna 15
    #--E_lim 40e3
    #--N_lim 40e3
    --track_time 8
    --n_times 96
    --t0 -4
    --f 1.4e9
    --df 0.0
    --n_freqs 1
    #--radio_array_seed 123
    --model_profile spergel
    --ell_prior_sigma 1.0
    --ell_prior_scale 0.2
    --g_prior_sigma 1.0
    --g_prior_scale 0.1
    --hlr_prior_sigma 1.0
    --hlr_prior_min 0.1
    --hlr_prior_max 3.0
    --flux_prior_sigma 1.0
    --flux_prior_min 0.03
    --flux_prior_max 0.25
    --lr_map 3e-3
    --n_steps_map 7000
    --sampler mclmc
    --n_warmup 70000
    --num_chains 10
    --num 20
    --num_steps 500
    --id run_${SLURM_ARRAY_TASK_ID}
    #--save_samples true
    --output_dir ${WORK}/repos/Radio_WL_Generative_Forward_Model/outputs/paper/shear_response/g_0_0
)

srun python -u shear_numpyro_sampling_argparse.py "${args[@]}"
