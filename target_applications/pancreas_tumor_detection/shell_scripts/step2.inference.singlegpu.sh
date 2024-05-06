datapath=/data/zzhou82/data/JHH_ROI_0.5mm
arch=$1
suprem_path=$2
savepath=./inference/$3
dataset_list=jhh_test

CUDA_VISIBLE_DEVICES=0 python -W ignore inference.py --save_dir $savepath --checkpoint $suprem_path --data_root_path $datapath --num_class 5 --map_type jhh --backbone $arch --dataset_list $dataset_list --a_min -100 --a_max 200 --stage test