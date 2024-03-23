#!/bin/bash
#SBATCH --job-name=inference

#SBATCH -N 1
#SBATCH -n 6
#SBATCH -G a100:1
##SBATCH --exclusive
#SBATCH --mem=80G
#SBATCH -p general
#SBATCH -t 3:00:00
#SBATCH -q public

#SBATCH -o %x_slurm_%j.out     
#SBATCH -e %xslurm_%j.err      
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=zzhou82@asu.edu

module load mamba/latest # only for Sol

# mamba create -n atlasinference python=3.9
source activate suprem

# pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
# pip install monai[all]==0.9.0
# pip install -r requirements.txt


### Inference SuPreM on the novel dataset
# savepath=/scratch/zzhou82/2024_0322/AbdomenAtlasDemoPredict
# datarootpath=/scratch/zzhou82/2024_0322/AbdomenAtlasDemo

# python -W ignore inference.py --save_dir $savepath.$1 --checkpoint $2 --data_root_path $datarootpath --backbone $1 --store_result --suprem

# # for backbone in unet; do for pretrainpath in ./pretrained_checkpoints/supervised_suprem_unet_2100.pth; do sbatch --error=logs/$backbone.inference.out --output=logs/$backbone.inference.out hg.sh $backbone $pretrainpath; done; done
# # for backbone in swinunetr; do for pretrainpath in ./pretrained_checkpoints/supervised_suprem_swinunetr_2100.pth; do sbatch --error=logs/$backbone.inference.out --output=logs/$backbone.inference.out hg.sh $backbone $pretrainpath; done; done


### Inference other AI on the novel dataset
pretrainpath=./pretrained_checkpoints/swin_unetr_totalsegmentator_vertebrae.pth
savepath=./AbdomenAtlasDemoPredict
datarootpath=/scratch/zzhou82/2024_0322/AbdomenAtlasDemo

python -W ignore inference.py --save_dir $savepath --checkpoint $pretrainpath --data_root_path $datarootpath --customize

# for datasetname in AbdomenAtlasDemo; do sbatch --error=logs/$datasetname.inference.vertebrae.out --output=logs/$datasetname.inference.vertebrae.out hg.sh; done