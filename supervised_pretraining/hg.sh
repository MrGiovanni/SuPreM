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

RANDOM_PORT=$((RANDOM % 64512 + 1024))
datapath=/scratch/zzhou82/data/AbdomenAtlas1.1Mini
wordembeddingpath=./pretrained_weights/txt_encoding_abdomenatlas1.1.pth
datasetversion=AbdomenAtlas1.1

### Training 

python -W ignore -m torch.distributed.launch --nproc_per_node=4 --master_port=$RANDOM_PORT train.py --dist True --data_root_path $datapath --num_workers 12 --log_name $datasetversion.$1 --pretrain $2 --word_embedding $wordembeddingpath --backbone $1 --lr 1e-4 --warmup_epoch 20 --batch_size 8 --max_epoch 800 --cache_dataset --num_class 25 --cache_num 20 --dataset_version $datasetversion

# for backbone in unet; do for pretrainpath in ./pretrained_weights/Genesis_Chest_CT.pt; do sbatch --error=logs/$backbone.out --output=logs/$backbone.out hg.sh $backbone $pretrainpath; done; done
# for backbone in swinunetr; do for pretrainpath in ./pretrained_weights/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt; do sbatch --error=logs/$backbone.out --output=logs/$backbone.out hg.sh $backbone $pretrainpath; done; done