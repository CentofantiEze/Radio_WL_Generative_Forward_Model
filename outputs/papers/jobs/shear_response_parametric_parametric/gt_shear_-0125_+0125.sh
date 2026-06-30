#!/bin/bash
#SBATCH --mail-user=ezequiel.centofanti@cea.fr
#SBATCH --mail-type=NONE
#SBATCH --job-name=-0125+0125
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH -C v100-32g
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=out_-0125+0125_%a.out
#SBATCH --error=err_-0125+0125_%a.err
#SBATCH -A prk@v100
#SBATCH --array=0-99

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
    --g1_true -0.0125
    --g2_true 0.0125
    --ell_scale 0.2
    --antenna_type file
    --antenna_file ../data/SKA-Mid.txt
    --track_time 8
    --n_times 96
    --t0 -4
    --f 1.4e9
    --df 0.0
    --n_freqs 1
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
    --output_dir ${WORK}/repos/Radio_WL_Generative_Forward_Model/outputs/paper/shear_response/g_-0125_+0125
)

srun python -u run.py "${args[@]}"
