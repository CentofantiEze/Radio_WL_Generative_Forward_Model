#!/bin/bash
#SBATCH --mail-user=ezequiel.centofanti@cea.fr
#SBATCH --mail-type=NONE
#SBATCH --job-name=test_shear_numpyro    # nom du job
#SBATCH --ntasks=1                   # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --ntasks-per-node=1          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:1                 # nombre de GPU par noeud (max 8 avec gpu_p2)
#SBATCH --cpus-per-task=10           # nombre de coeurs CPU par tache (un quart du noeud ici)
#SBATCH -C v100-32g
#SBATCH --hint=nomultithread          # hyperthreading desactive
#SBATCH --time=02:00:00               # temps d'execution maximum demande (HH:MM:SS)
#SBATCH --output=out_sersic_shear_numpyro.out   # nom du fichier de sortie
#SBATCH --error=err_sersic_shear_numpyro.err    # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH -A prk@v100                   # specify the project

# nettoyage des modules charges en interactif et herites par defaut
module purge

# chargement des modules
module load tensorflow-gpu/py3/2.16.1

# echo launched commands
set -x

cd ${WORK}/repos/Radio_WL_Generative_Forward_Model/scripts

srun python sersic_shear_numpyro_sampling_argparse.py \
    --Ngal 100\
    --Npx 128\
    --pixel_scale 0.15\
    --noise_uv 0.004\
    --params_dir ../data/trecs_gal_params.npy\
    --g1_true -0.05\
    --g2_true 0.05\
    --ell_sigma 1.0\
    --ell_scale 0.3\
    --g_sigma 1.0\
    --g_scale 0.3\
    --sersic_index 1.0\
    --n_antenna 50\
    --E_lim 50e3\
    --N_lim 50e3\
    --track_time 10\
    --n_times 4\
    --f 1.4e9\
    --df 1e8\
    --n_freqs 4\
    --radio_array_seed 123\
    --ell_prior_sigma 1.0\
    --ell_prior_scale 0.3\
    --g_prior_sigma 1.0\
    --g_prior_scale 0.3\
    --hlr_prior_sigma 1.0\
    --hlr_prior_min 0.1\
    --hlr_prior_max 3.0\
    --flux_prior_sigma 1.0\
    --flux_prior_min 0.03\
    --flux_prior_max 0.25\
    --lr_map 3e-3\
    --n_steps_map 5000\
    --n_warmup 5000\
    --num_chains 10\
    --step_size 0.005\
    --num 20\
    --num_steps 10000
