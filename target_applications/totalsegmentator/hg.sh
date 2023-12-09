#!/bin/bash
#SBATCH --job-name=suprem

#SBATCH -N 1
#SBATCH -n 12
#SBATCH -G a100:1
##SBATCH --exclusive
#SBATCH --mem=100G
#SBATCH -p general
#SBATCH -t 0-05:00:00
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

# cd pretrained_weights/
# wget https://www.dropbox.com/scl/fi/gd1d7k9mac5azpwurds66/supervised_suprem_swinunetr_2100.pth?rlkey=xoqr7ey52rnese2k4hwmrlqrt
# mv supervised_suprem_swinunetr_2100.pth\?rlkey\=xoqr7ey52rnese2k4hwmrlqrt supervised_suprem_swinunetr_2100.pth
# cd ../

RANDOM_PORT=$((RANDOM % 64512 + 1024))
datapath=/scratch/zzhou82/data/Totalsegmentator_dataset/Totalsegmentator_dataset/ 
# change to /path/to/your/data/TotalSegmentator
arch=swinunetr 
# support swinunetr, unet, and segresnet
target_task=vertebrae
num_target_class=25
num_target_annotation=64
suprem_path=pretrained_weights/supervised_suprem_swinunetr_2100.pth
checkpoint_path=out/efficiency.$arch.$target_task.number$num_target_annotation/best_model.pth

python -W ignore -m torch.distributed.launch --nproc_per_node=1 --master_port=$RANDOM_PORT train.py --dist False --model_backbone $arch --log_name efficiency.$arch.$target_task.number$num_target_annotation --map_type $target_task --num_class $num_target_class --dataset_path $datapath --num_workers 12 --batch_size 8 --pretrain $suprem_path --percent $num_target_annotation

# sbatch --error=logs/train.out --output=logs/train.out hg.sh

# python -W ignore -m torch.distributed.launch --nproc_per_node=1 --master_port=$RANDOM_PORT test.py --dist False --model_backbone $arch --log_name efficiency.$arch.$target_task.number$num_target_annotation --map_type $target_task --num_class $num_target_class --dataset_path $datapath --num_workers 12 --batch_size 2 --pretrain $checkpoint_path --train_type efficiency 

# # sbatch --error=logs/test.out --output=logs/test.out hg.sh