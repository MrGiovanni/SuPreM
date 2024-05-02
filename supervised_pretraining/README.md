# Pre-training SuPreM on the AbdomenAtlas 1.1 dataset

##### 0. Create a virtual environment (optional)

```bash
conda create -n suprem python=3.8
source activate suprem
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install monai[all]==0.9.0
pip install -r requirements.txt
```

##### 1. Clone and setup the GitHub repository

```bash
git clone https://github.com/MrGiovanni/SuPreM
cd SuPreM/supervised_pretraining/pretrained_weights/
wget https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt
wget https://www.dropbox.com/s/lh5kuyjxwjsxjpl/Genesis_Chest_CT.pt
cd ..
```

##### 2. Pre-train models on AbdomenAtlas 1.1

```bash

RANDOM_PORT=$((RANDOM % 64512 + 1024))
datapath=/scratch/zzhou82/data/AbdomenAtlas1.1Mini
backbone=unet # or swinunetr
pretrainpath=./pretrained_weights/Genesis_Chest_CT.pt
# pretrainpath=./pretrained_weights/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt # for swinunetr
datasetversion=AbdomenAtlas1.1 # or AbdomenAtlas1.0
wordembeddingpath=./pretrained_weights/txt_encoding_abdomenatlas1.1.pth
# wordembeddingpath=./pretrained_weights/txt_encoding_abdomenatlas1.0.pth # for AbdomenAtlas 1.0

python -W ignore -m torch.distributed.launch --nproc_per_node=4 --master_port=$RANDOM_PORT train.py --dist  --data_root_path $datapath --num_workers 12 --log_name $datasetversion.$backbone --pretrain $pretrainpath --word_embedding $wordembeddingpath --backbone $backbone --lr 1e-4 --warmup_epoch 20 --batch_size 8 --max_epoch 800 --cache_dataset --num_class 25 --cache_num 150 --dataset_version $datasetversion
```
