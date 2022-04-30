#!/bin/bash

## change the last two digits to your team id
#SBATCH --account=csci_ga_2572_2022sp_01

## change the partition number to use different number of GPUs
##SBATCH --partition=n1s8-v100-1
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=8

##SBATCH --partition=n1s16-v100-2
##SBATCH --gres=gpu:2
##SBATCH --cpus-per-task=16

#SBATCH --partition=n1c24m128-v100-4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
##SBATCH --mem-per-cpu=8G


#SBATCH --time=24:00:00
#SBATCH --output=swavdp_%j.out
#SBATCH --error=swavdp_%j.err
#SBATCH --exclusive
#SBATCH --requeue


mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

cp -rp /scratch/DL22SP/unlabeled_224.sqsh /tmp
cp -rp /scratch/DL22SP/labeled.sqsh /tmp
echo "Dataset is copied to /tmp"



singularity exec --nv \
--bind /scratch \
--overlay /scratch/DL22SP/conda.ext3:ro \
--overlay /tmp/unlabeled_224.sqsh \
--overlay /tmp/labeled.sqsh \
/share/apps/images/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
/bin/bash -c "
source /ext3/env.sh
conda activate
python swav_dp.py \
--workers 2 \
--epochs 100 \
--base_lr 0.01 \
--final_lr 0.00001 \
--epsilon 0.05 \
--warmup_epochs 10 \
--start_warmup 0.3 \
--batch_size 256 \
--size_crops 224 96 \
--nmb_crops 2 6 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--freeze_prototypes_niters 5005 \
--queue_length 2048 \
--epoch_queue_starts 30 \
--nmb_prototypes 1500 \
--checkpoint_freq 5 \
--arch resnet50 \
--temperature 0.1 \
--trial 2048
"

