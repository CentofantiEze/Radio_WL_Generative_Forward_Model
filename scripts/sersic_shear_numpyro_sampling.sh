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

srun python sersic_shear_numpyro_sampling.py