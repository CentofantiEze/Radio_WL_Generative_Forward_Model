#!/bin/bash
#SBATCH --mail-user=ezequiel.centofanti@cea.fr
#SBATCH --mail-type=NONE
#SBATCH --job-name=coverage_testing
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH -C v100-32g
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=logs/coverage_0_%a.out
#SBATCH --error=logs/coverage_0_%a.err
#SBATCH -A prk@v100
#SBATCH --array=0-9999

# Coverage testing: N_COV coverage tests x N_SUB sub-runs each
# Total jobs = N_COV * N_SUB
# 2D indexing from (N_BASE + SLURM_ARRAY_TASK_ID):
#   global_id   = N_BASE + TASK_ID
#   coverage_id = global_id / N_SUB
#   sub_run_id  = global_id % N_SUB
#
# To extend coverage runs, set N_BASE to the next batch start:
#   Batch 1: N_BASE=0,     --array=0-9999   -> cov_0   to cov_99
#   Batch 2: N_BASE=10000, --array=0-9999   -> cov_100 to cov_199
#   Batch 3: N_BASE=20000, --array=0-9999   -> cov_200 to cov_299
#   ...
#   Batch 10: N_BASE=90000, --array=0-9999   -> cov_900 to cov_999

N_BASE=0
N_COV=100
N_SUB=100

GLOBAL_ID=$((N_BASE + SLURM_ARRAY_TASK_ID))
COVERAGE_ID=$((GLOBAL_ID / N_SUB))
SUB_RUN_ID=$((GLOBAL_ID % N_SUB))

# No --seed passed: each sub-run picks a random seed automatically,
# so each generates a different set of 100 galaxies.
# Combining 100 sub-runs gives 10,000 independent galaxy measurements.

module purge
module load anaconda-py3/2024.06
conda activate forwardmodel

set -x

cd ${WORK}/repos/Radio_WL_Generative_Forward_Model/scripts

OUTPUT_DIR=${WORK}/repos/Radio_WL_Generative_Forward_Model/outputs/paper/coverage/cov_${COVERAGE_ID}

args=(
    --Ngal 100
    --Npx 128
    --pixel_scale 0.15
    --noise_uv 0.02
    --trecs_data_path ../data/trecs_gal_params.npy
    --data_profile spergel
    --g1_true -0.05
    --g2_true 0.05
    --ell_scale 0.2
    --antenna_type file
    --antenna_file ../data/SKA-Mid.txt
    --track_time 8
    --n_times 96
    --t0 -4
    --f 1.4e9
    --df 1e8
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
    --id mclmc_spergel_${SUB_RUN_ID}
    --output_dir ${OUTPUT_DIR}
)

srun python -u run.py "${args[@]}"
