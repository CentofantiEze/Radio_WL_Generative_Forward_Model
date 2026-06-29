#!/bin/bash
#SBATCH --mail-user=ezequiel.centofanti@cea.fr
#SBATCH --mail-type=NONE
#SBATCH --job-name=spergel_cosmos
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH -C v100-32g
#SBATCH --hint=nomultithread
#SBATCH --time=01:59:00
#SBATCH --output=out_spergel_cosmos_%a.out
#SBATCH --error=err_spergel_cosmos_%a.err
#SBATCH -A prk@v100
#SBATCH --qos=qos_gpu-t3
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
    --noise_uv 0.01
    --noise_data 0.01
    --trecs_data_path ../data/trecs_gal_params.npy
    --cosmos_data_path /lustre/fsn1/projects/rech/prk/uds36vp/datasets/COSMOS_25.2_training_sample
    --cosmos_sample 25.2
    --mag_cut 24.0
    --data_profile VAE
    --data_vae_path /lustre/fsn1/projects/rech/prk/uds36vp/pshear/cosmos/runs/pretrained_benjamin
    --data_vae_epoch 920
    --g1_true -0.05
    --g2_true 0.05
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
    --hlr_prior_max 5.0
    --flux_prior_sigma 1.0
    --flux_prior_min 0.005
    --flux_prior_max 1.
    --lr_map 3e-3
    --n_steps_map 3000
    --sampler mclmc
    --n_warmup 70000
    --num_chains 10
    --num 10
    --num_steps 500
    --id single_run_${SLURM_ARRAY_TASK_ID}
    --save_samples
    --save_plots
    --save_data
    --output_dir ${WORK}/repos/Radio_WL_Generative_Forward_Model/outputs/paper/parametric_hst_whitened
)

srun python -u run.py "${args[@]}"
