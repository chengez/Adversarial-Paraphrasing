#!/bin/bash

#SBATCH --job-name=create_uniwmmage
#SBATCH --output=logging/logging.%j.create_uniwmmage
#SBATCH --time=24:00:00             #Set the wall clock limit to 24hr
#SBATCH --ntasks=1                  #Request 1 task
#SBATCH --ntasks-per-node=1         #Request 1 task/core per node
#SBATCH --mem=32gb                  #Request 64GB memory
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtxa6000:2           #Request 1 GPU per node can be 1 or 2

#SBATCH --qos=cml-scavenger
#SBATCH --partition=cml-scavenger
#SBATCH --account=cml-scavenger


# Load necessary modules and activate the environment
source /fs/nexus-scratch/yzcheng/anaconda3/etc/profile.d/conda.sh
conda activate advpara

cd /fs/cml-scratch/yzcheng/Adversarial-Paraphrasing

wm_name="kgw_wm"

python ${wm_name}/create_watermarked_mage.py --save_path ${wm_name}/wm_mage
