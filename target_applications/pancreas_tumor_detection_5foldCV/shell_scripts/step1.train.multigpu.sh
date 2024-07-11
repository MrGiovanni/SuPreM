# STEP I. Train

RANDOM_PORT=$((RANDOM % 64512 + 1024))
datapath=/mnt/ccvl15/zzhou82/data/JHH_ROI_0.5mm
arch=$1
suprem_path=$2
# resume_path=out/$arch.jhh.fold_$6/checkpoint_epoch_200.pth 
num_target_class=5
batch_size=$3
num_samples=$4
num_workers=12
dataset_list=$5
fold_id=$6 

# Multiple GPUs
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 python -W ignore -m torch.distributed.launch --nproc_per_node=6 --master_port=$RANDOM_PORT train.py --dist --model_backbone $arch --log_name $arch.jhh.fold_$fold_id --num_class $num_target_class --dataset_path $datapath --num_workers $num_workers --batch_size $batch_size --pretrain $suprem_path --lr 5e-4 --dataset_list $dataset_list --num_samples $num_samples --a_min -100 --a_max 200 --max_epoch 1000 --stage train  
#--resume $resume_path
