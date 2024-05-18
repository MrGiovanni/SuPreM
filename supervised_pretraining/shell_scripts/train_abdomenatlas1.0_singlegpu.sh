#!/bin/bash
#SBATCH --job-name=abdomenatlas1.0-singlegpu

#SBATCH -N 1
#SBATCH -n 12
#SBATCH -G a100:1
##SBATCH --exclusive
#SBATCH --mem=100G
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

# ### Training (AbdomenAtlas 1.0) 
if [ "$1" = "segresnet" ]; then
    batch_size=16
elif [ "$1" = "unet" ]; then
    batch_size=8
elif [ "$1" = "swinunetr" ]; then
    batch_size=2
fi

nproc_per_node=1
num_workers=$((12 * nproc_per_node))
cache_num=100
RANDOM_PORT=$((RANDOM % 64512 + 1024))
datapath=/scratch/zzhou82/data/AbdomenAtlas1.0Mini
datasetversion=AbdomenAtlas1.0 # or AbdomenAtlas1.0
wordembeddingpath=./pretrained_weights/txt_encoding_abdomenatlas1.0.pth # for AbdomenAtlas 1.0

# Single GPUs
python -W ignore -m torch.distributed.launch --nproc_per_node=$nproc_per_node --master_port=$RANDOM_PORT train.py --data_root_path $datapath --dataset_list $datasetversion --num_workers $num_workers --log_name $datasetversion.$1.singlegpu --word_embedding $wordembeddingpath --backbone $1 --lr 1e-3 --warmup_epoch 20 --batch_size $batch_size --max_epoch 2000 --cache_dataset --num_class 9 --cache_num $cache_num --dataset_version $datasetversion