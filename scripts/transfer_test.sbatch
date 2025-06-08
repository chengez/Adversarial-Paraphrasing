#!/bin/bash

#SBATCH --job-name=roblarge-gltr
#SBATCH --output=logging/logging.%j.roblarge-gltr
#SBATCH --time=24:00:00             #Set the wall clock limit to 24hr
#SBATCH --ntasks=1                  #Request 1 task
#SBATCH --ntasks-per-node=1         #Request 1 task/core per node
#SBATCH --mem=16gb                  #Request 64GB memory
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtxa6000:2          #Request 1 GPU per node can be 1 or 2

#SBATCH --qos=medium


# Load necessary modules and activate the environment
source /fs/nexus-scratch/yzcheng/anaconda3/etc/profile.d/conda.sh
conda activate advpara

# Change to the project directory
cd /fs/cml-scratch/yzcheng/Adversarial-Paraphrasing

DEBUG=0
### Modify ##############################################
# adversarial=0
adversarial=1

dataset='mage'
# dataset='kgwwm_mage'
# dataset='uniwm_mage'

# guidance_classifier='openai_roberta_large'
# guidance_classifier='openai_roberta_base'
guidance_classifier='radar'
# guidance_classifier='mage'

# deploy_classifier='mage'
# deploy_classifier='kgw_wm'
deploy_classifier='uni_wm'
# deploy_classifier='rdf_wm'
# deploy_classifier='openai_roberta_large'
# deploy_classifier='openai_roberta_base'
# deploy_classifier='radar'
# deploy_classifier='gltr'
# deploy_classifier='fastdetectgpt'
#########################################################

python paraphrase_and_detect.py \
    --dataset ${dataset} \
    --deploy_classifier ${deploy_classifier} \
    --guidance_classifier ${guidance_classifier} \
    --top_p 0.99 \
    --num_samples 2000 \
    --batch_size 5 \
    --adversarial ${adversarial} \
    --debug ${DEBUG}
