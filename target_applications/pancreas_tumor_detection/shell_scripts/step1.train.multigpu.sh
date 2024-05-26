# STEP I. Train

RANDOM_PORT=$((RANDOM % 64512 + 1024))
datapath=/data/zzhou82/data/JHH_ROI_0.5mm
arch=$1
suprem_path=$2
resume_path=out/efficiency.segresnet.jhh/checkpoint_epoch_200.pth # change to model.pth for resuming from the current model  
num_target_class=5
batch_size=$3
num_samples=$4
num_workers=10
dataset_list=jhh_train # change to jhh_train_mini for pdac&cyst only in venous and pnet only in arterial

# Multiple GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -W ignore -m torch.distributed.launch --nproc_per_node=8 --master_port=$RANDOM_PORT train.py --dist --model_backbone $arch --log_name $arch.jhh --num_class $num_target_class --dataset_path $datapath --num_workers $num_workers --batch_size $batch_size --pretrain $suprem_path --lr 5e-4 --dataset_list $dataset_list --num_samples $num_samples --a_min -100 --a_max 200 --max_epoch 1000 --stage train
# --resume $resume_path
