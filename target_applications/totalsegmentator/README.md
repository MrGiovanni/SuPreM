<h1 align="center">SuPreM</h1>
<h3 align="center" style="font-size: 20px; margin-bottom: 4px">For Anatomical Segmentation on the (subset of) TotalSegmentator</h3>
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

##### 1. Clone the GitHub repository

```bash
git clone https://github.com/MrGiovanni/SuPreM
cd SuPreM/
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install monai[all]==0.9.0
pip install pip==23.3.1
pip install -r requirements.txt
```

##### 2. Download the pre-trained Swin UNETR checkpoint

```bash
cd target_applications/totalsegmentator/pretrained_weights/
wget https://huggingface.co/MrGiovanni/SuPreM/resolve/main/supervised_suprem_swinunetr_2100.pth
wget https://huggingface.co/MrGiovanni/SuPreM/resolve/main/supervised_suprem_segresnet_2100.pth
wget https://huggingface.co/MrGiovanni/SuPreM/resolve/main/supervised_suprem_unet_2100.pth
cd ../../../
```

##### 3. Download the TotalSegmentator dataset

from [Zenodo](https://doi.org/10.5281/zenodo.6802613) (1,228 subjects) and save it to `/path/to/your/data/TotalSegmentator`

##### 4. Fine-tune the pre-trained Swin UNETR on TotalSegmentator

```bash
# Single GPU

cd target_applications/totalsegmentator/
RANDOM_PORT=$((RANDOM % 64512 + 1024))
datapath=/scratch/zzhou82/data/Totalsegmentator_dataset/Totalsegmentator_dataset/ # change to /path/to/your/data/TotalSegmentator
arch=swinunetr # support swinunetr, unet, and segresnet
suprem_path=pretrained_weights/supervised_suprem_swinunetr_2100.pth
target_task=vertebrae
num_target_class=25
num_target_annotation=64

python -W ignore -m torch.distributed.launch --nproc_per_node=1 --master_port=$RANDOM_PORT train.py --dist  --model_backbone $arch --log_name efficiency.$arch.$target_task.number$num_target_annotation --map_type $target_task --num_class $num_target_class --dataset_path $datapath --num_workers 8 --batch_size 2 --pretrain $suprem_path --percent $num_target_annotation
```

##### 5. Evaluate the performance per class

```bash
# Single GPU

cd target_applications/totalsegmentator/
RANDOM_PORT=$((RANDOM % 64512 + 1024))
datapath=/scratch/zzhou82/data/Totalsegmentator_dataset/Totalsegmentator_dataset/ # change to /path/to/your/data/TotalSegmentator
checkpoint_path=out/efficiency.$arch.$target_task.number$num_target_annotation/best_model.pth
target_task=vertebrae
num_target_class=25
num_target_annotation=64

python -W ignore -m torch.distributed.launch --nproc_per_node=1 --master_port=$RANDOM_PORT test.py --dist  --model_backbone $arch --log_name efficiency.$arch.$target_task.number$num_target_annotation --map_type $target_task --num_class $num_target_class --dataset_path $datapath --num_workers 8 --batch_size 2 --pretrain $checkpoint_path --train_type efficiency 
```
