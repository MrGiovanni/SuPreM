# Single GPU

cd /home/chen/EMoT/target_applications/ImageCAS
RANDOM_PORT=$((RANDOM % 64512 + 1024))
pretraining_method_name=med3d
datapath=/home/chen/ImageCAS/ # change to /path/to/your/data/ImageCAS
target_task=cas
num_target_class=2
for arch in unet; do
    if [ "$arch" = "unet" ]; then
    case $pretraining_method_name in 
    suprem)
        pretrain_path="pretrained_weights/supervised_suprem_unet_2100.pth"
        ;;
    genesis)
        pretrain_path="pretrained_weights/self_supervised_models_genesis_unet_620.pt"
        ;;
    med3d)
        pretrain_path="pretrained_weights/supervised_med3D_residual_unet_1623.pth"
        ;;
    scratch)
        echo "from scratch"
        ;;
    *)
        echo "unkown: $pretraining_method_name"
        ;;
    esac

    elif [ "$arch" = "segresnet" ]; then
    case $pretraining_method_name in 
    suprem)
        pretrain_path="pretrained_weights/supervised_suprem_segresnet_2100.pth"
        ;;
    scratch)
        echo "from scratch"
        ;;
    *)
        echo "unkown: $pretraining_method_name"
        ;;
    esac
        
    else
        echo "unkown : $arch"
    fi

    for fold in 1; do
        RANDOM_PORT=$((RANDOM % 64512 + 1024))
        python -W ignore -m torch.distributed.launch --nproc_per_node=1 --master_port=$RANDOM_PORT train.py   --model_backbone $arch --log_name $pretraining_method_name.$arch.$target_task.fold$fold --map_type $target_task --num_class $num_target_class --dataset_path $datapath --num_workers 8 --batch_size 2 --pretrain $pretrain_path --fold $fold --pretraining_method_name $pretraining_method_name
    done
done