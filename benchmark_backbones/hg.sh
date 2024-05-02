#!/bin/bash
#SBATCH --job-name=backbone

#SBATCH -N 1
#SBATCH -n 48
#SBATCH -G a100:1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -p general
#SBATCH -t 7-00:00:00
#SBATCH -q public

#SBATCH -o %x_slurm_%j.out     
#SBATCH -e %xslurm_%j.err      
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=zzhou82@asu.edu

module load mamba/latest # only for Sol

# mamba create -n suprem python=3.9
source activate suprem

# pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
# pip install monai[all]==0.9.0
# pip install -r requirements.txt

backbone=unet
RANDOM_PORT=$((RANDOM % 64512 + 1024))
datapath=/scratch/zzhou82/data/AbdomenAtlasMini1.0

### Training 

python -W ignore -m torch.distributed.launch --nproc_per_node=4 --master_port=$RANDOM_PORT train.py --dist  --data_root_path $datapath --num_workers 12 --log_name AbdomenAtlas1.0.$backbone --backbone $backbone --lr 1e-4 --warmup_epoch 20 --batch_size 2 --max_epoch 800 --cache_dataset

# sbatch --error=logs/unet.out --output=logs/unet.out hg.sh