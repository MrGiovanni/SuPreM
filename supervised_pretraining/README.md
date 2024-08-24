<h1 align="center">SuPreM</h1>
<h3 align="center" style="font-size: 20px; margin-bottom: 4px">Pre-Train on AbdomenAtlas 1.1</h3>
<p align="center">
    <a href='https://www.zongweiz.com/dataset'><img src='https://img.shields.io/badge/Project-Page-Green'></a> 
    <a href='https://www.cs.jhu.edu/~alanlab/Pubs23/li2023suprem.pdf'><img src='https://img.shields.io/badge/Paper-PDF-purple'></a> 
    <a href='document/promotion_slides.pdf'><img src='https://img.shields.io/badge/Slides-PDF-orange'></a> 
    <a href='document/dom_wse_poster.pdf'><img src='https://img.shields.io/badge/Poster-PDF-blue'></a> 
    <a href='https://www.cs.jhu.edu/news/ai-and-radiologists-unite-to-map-the-abdomen/'><img src='https://img.shields.io/badge/WSE-News-yellow'></a>
    <br/>
    <a href="https://github.com/MrGiovanni/SuPreM"><img src="https://img.shields.io/github/stars/MrGiovanni/SuPreM?style=social" /></a>
    <a href="https://twitter.com/bodymaps317"><img src="https://img.shields.io/twitter/follow/BodyMaps" alt="Follow on Twitter" /></a>
</p>

##### 0. Create a virtual environment (optional)

<details>
<summary style="margin-left: 25px;">[Optional] Install Anaconda on Linux</summary>
<div style="margin-left: 25px;">
    
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
bash Anaconda3-2024.06-1-Linux-x86_64.sh -b -p ./anaconda3
./anaconda3/bin/conda init
source ~/.bashrc
```
</div>
</details>

```bash
conda create -n suprem python=3.8 -y
source activate suprem
```

##### 1. Clone and setup the GitHub repository

```bash
git clone https://github.com/MrGiovanni/SuPreM
cd SuPreM/
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install monai[all]==0.9.0
pip install pip==23.3.1
pip install -r requirements.txt
cd ./supervised_pretraining/pretrained_weights/
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
num_class_in_dataset=25 # or 9
wordembeddingpath=./pretrained_weights/txt_encoding_abdomenatlas1.1.pth
# wordembeddingpath=./pretrained_weights/txt_encoding_abdomenatlas1.0.pth # for AbdomenAtlas 1.0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -W ignore -m torch.distributed.launch --nproc_per_node=8 --master_port=$RANDOM_PORT train.py --dist --dataset_list $datasetversion --data_root_path $datapath --num_workers 10 --log_name $datasetversion.$backbone --pretrain $pretrainpath --word_embedding $wordembeddingpath --backbone $backbone --lr 1e-3 --warmup_epoch 20 --batch_size 8 --max_epoch 800 --cache_dataset --num_class $num_class_in_dataset --cache_num 300 --dataset_version $datasetversion
```

If you want to pre-train a model with SegResNet backbone (no publicly available pre-trained weights), please use the following command:
```bash
RANDOM_PORT=$((RANDOM % 64512 + 1024))
datapath=/scratch/zzhou82/data/AbdomenAtlas1.1Mini
backbone=segresnet
backbone_initial_filters=16 # or 32 for larger model
datasetversion=AbdomenAtlas1.1 # or AbdomenAtlas1.0
num_class_in_dataset=25 # or 9

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -W ignore -m torch.distributed.launch --nproc_per_node=8 --master_port=$RANDOM_PORT train.py --dist --dataset_list $datasetversion --data_root_path $datapath --num_workers 10 --log_name $datasetversion.$backbone --backbone $backbone --segresnet_init_filters $backbone_initial_filters --lr 1e-3 --warmup_epoch 20 --batch_size 16 --max_epoch 800 --cache_dataset --num_class $num_class_in_dataset --cache_num 300 --dataset_version $datasetversion
```
