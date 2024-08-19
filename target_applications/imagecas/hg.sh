#!/bin/bash
#SBATCH --job-name=suprem-total

#SBATCH -N 1
#SBATCH -n 8
#SBATCH -G a100:1
##SBATCH --exclusive
#SBATCH --mem=80G
#SBATCH -p general
#SBATCH -t 2-00:00:00
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
# datapath=/scratch/zzhou82/data/Totalsegmentator_dataset/Totalsegmentator_dataset/ 
datapath=/path/to/your/data/imagecas/
# change to /path/to/your/data/TotalSegmentator
arch=$1 
# support swinunetr, unet, and segresnet
target_task=$2
num_target_class=$3
fold=$4
pretraining_method_name=$5
# the maximum number of target annotations is 1081 for the whole training dataset
if [ "$arch" = "unet" ]; then case $pretraining_method_name in suprem) pretrain_path="pretrained_weights/supervised_suprem_unet_2100.pth" ;; genesis) pretrain_path="pretrained_weights/self_supervised_models_genesis_unet_620.pt" ;; scratch) echo "from scratch" ;; *) echo "unkown: $pretraining_method_name" ;; esac; elif [ "$arch" = "segresnet" ]; then case $pretraining_method_name in suprem) pretrain_path="pretrained_weights/supervised_suprem_segresnet_2100.pth" ;; scratch) echo "from scratch" ;; *) echo "unkown: $pretraining_method_name" ;; esac; fi
log_name=$pretraining_method_name.$arch.$target_task.fold$fold
checkpoint_path=checkpoints/$pretraining_method_name.$arch.$target_task.fold$fold/best_model.pth

### Training 
if [ "$pretraining_method_name" == "scratch" ]; then python -W ignore -m torch.distributed.launch --nproc_per_node=1 --master_port=$RANDOM_PORT train.py   --model_backbone $arch --log_name $log_name --map_type $target_task --num_class $num_target_class --dataset_path $datapath --num_workers 8 --batch_size 2  --fold $fold --pretraining_method_name $pretraining_method_name; else python -W ignore -m torch.distributed.launch --nproc_per_node=1 --master_port=$RANDOM_PORT train.py   --model_backbone $arch --log_name $log_name --map_type $target_task --num_class $num_target_class --dataset_path $datapath --num_workers 8 --batch_size 2  --fold $fold --pretraining_method_name $pretraining_method_name --pretrain $pretrain_path; fi

# for pretraining_method_name in suprem scratch; do for arch in segresnet; do for fold in 1; do sbatch --error=logs/train.$pretraining_method_name.$arch.vertebrae.fold$fold.out --output=logs/train.$pretraining_method_name.$arch.vertebrae.fold$fold.out hg.sh  $arch cas 2 $fold $pretraining_method_name; done; done; done


### Testing
python -W ignore -m torch.distributed.launch --nproc_per_node=1 --master_port=$RANDOM_PORT test.py   --model_backbone $arch --log_name $log_name --map_type $target_task --num_class $num_target_class --dataset_path $datapath --num_workers 8 --batch_size 2 --pretrain $checkpoint_path  --fold $fold --pretraining_method_name $pretraining_method_name

# for pretraining_method_name in suprem scratch; do for arch in segresnet; do for fold in 1; do sbatch --error=logs/eval.$pretraining_method_name.$arch.vertebrae.fold$fold.out --output=logs/eval.$pretraining_method_name.$arch.vertebrae.fold$fold.out hg.sh  $arch cas 2 $fold $pretraining_method_name; done; done; done
