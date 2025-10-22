#!/bin/bash
#SBATCH --mail-user=ezequiel.centofanti@cea.fr
#SBATCH --mail-type=NONE
#SBATCH --job-name=spergel_run    # nom du job
#SBATCH --ntasks=1                   # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --ntasks-per-node=1          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:1                 # nombre de GPU par noeud (max 8 avec gpu_p2)
#SBATCH --cpus-per-task=10           # nombre de coeurs CPU par tache (un quart du noeud ici)
#SBATCH -C v100-32g
#SBATCH --hint=nomultithread          # hyperthreading desactive
#SBATCH --time=10:00:00               # temps d'execution maximum demande (HH:MM:SS)
#SBATCH --output=out_spergel_run_%a.out   # nom du fichier de sortie
#SBATCH --error=err_spergel_run_%a.err    # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH -A prk@v100                   # specify the project
#SBATCH --array=0-99                 # array job with 10 tasks

# nettoyage des modules charges en interactif et herites par defaut
module purge

# chargement des modules
module load tensorflow-gpu/py3/2.16.1

# echo launched commands
set -x

cd ${WORK}/repos/Radio_WL_Generative_Forward_Model/scripts

args=(
    --Ngal 100
    --Npx 128
    --pixel_scale 0.15
    --noise_uv 0.004 
    --trecs_data_path ../data/trecs_gal_params.npy 
    # --deepshape_data_path ../data/val_set_rivi.h5
    # --cosmos_data_path /lustre/fsn1/projects/rech/prk/uds36vp/datasets/COSMOS_23.5_training_sample
    --data_profile spergel
    # --sersic_index 1.0 
    --g1_true -0.05 
    --g2_true 0.05 
    --ell_sigma 1.0 
    --ell_scale 0.2 
    --g_sigma 1.0 
    --g_scale 0.1
    --antenna_type random
    # --antenna_file ../data/SKA-Mid.txt
    # --uv_mask_weighting histogram
    --n_antenna 15 
    --E_lim 40e3 
    --N_lim 40e3 
    --track_time 8 
    --n_times 96 
    --t0 -4 
    --f 1.4e9 
    --df 1e8 
    --n_freqs 1 
    --radio_array_seed 123 
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
    --n_steps_map 2500 
    --n_warmup 50 
    --num_chains 10 
    # --step_size 0.005 
    --num 20 
    --num_steps 5000 
    --id spergel_run_${SLURM_ARRAY_TASK_ID} 
    # --save_samples false
    --plot_chains scaled
    --output_dir /lustre/fswork/projects/rech/prk/uds36vp/repos/Radio_WL_Generative_Forward_Model/outputs/spergel_parallel_100
)

srun python shear_numpyro_sampling_argparse.py "${args[@]}"
  
