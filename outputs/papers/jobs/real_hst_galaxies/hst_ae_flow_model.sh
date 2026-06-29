#!/bin/bash
#SBATCH --mail-user=ezequiel.centofanti@cea.fr
#SBATCH --mail-type=NONE
#SBATCH --job-name=100flowmclmc
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH -C v100-32g
#SBATCH --hint=nomultithread
#SBATCH --time=01:59:00
#SBATCH --output=out_tunning_%a.out
#SBATCH --error=err_tunning_%a.err
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
    --mag_cut 24.
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
    --df 1e8
    --n_freqs 1
    --model_profile VAE
    --g_prior_sigma 1.
    --g_prior_scale 0.1
    --vae_epoch 500
    #--vae_path /lustre/fsn1/projects/rech/prk/uds36vp/pshear/cosmos/runs/distill_half_s3beml4e
    --vae_path /lustre/fsn1/projects/rech/prk/uds36vp/pshear/cosmos/runs/distill_quarter_9rngz71t
    --vae_model_inference_mode parallel
    #--vae_precision float32
    --latent_dim 4
    --pixel_scale_vae 0.03
    --latent_sigma 1.
    --use_flow
    --flow_path /lustre/fsn1/projects/rech/prk/uds36vp/pshear/cosmos/runs/flow_nvp_uncond_fast/13ciac4j
    --flow_epoch 5000
    --lr_map 3e-2
    --n_steps_map 2000
    --sampler mclmc
    --n_warmup 5000
    --num_chains 4
    --num 5
    --num_steps 500
    --id run_${SLURM_ARRAY_TASK_ID}
    --save_samples
    --save_plots
    --save_data
    --output_dir /lustre/fswork/projects/rech/prk/uds36vp/repos/Radio_WL_Generative_Forward_Model/outputs/paper/ae_white_combined
)

srun python -u run.py "${args[@]}"
