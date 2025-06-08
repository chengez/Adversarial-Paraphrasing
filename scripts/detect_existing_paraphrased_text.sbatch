#!/bin/bash

#SBATCH --job-name=radar-roblarge
#SBATCH --output=logging/logging.%j.radar-roblarge
#SBATCH --time=10:00:00             #Set the wall clock limit to 24hr
#SBATCH --ntasks=1                  #Request 1 task
#SBATCH --ntasks-per-node=1         #Request 1 task/core per node
#SBATCH --mem=32gb                  #Request 64GB memory
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtxa6000:1           #Request 1 GPU per node can be 1 or 2

#SBATCH --qos=default


# Load necessary modules and activate the environment
source /fs/nexus-scratch/yzcheng/anaconda3/etc/profile.d/conda.sh
conda activate advpara

cd /fs/cml-scratch/yzcheng/Adversarial-Paraphrasing


python detect_existing_paraphrased_text.py \
    --paraphrased_texts_path outputs/guided_generations_mage/adv/radar \
    --deploy_classifier openai_roberta_large



