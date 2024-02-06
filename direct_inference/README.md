# Predict organ masks on unseen CT volumes using SuPreM (direct inference)

### 1. Getting started
##### 1.1 Installation

```bash
conda create -n suprem python=3.8
source activate suprem

git clone https://github.com/MrGiovanni/SuPreM
cd SuPreM/
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install monai[all]==0.9.0
pip install -r requirements.txt
```

##### 1.2 Download the datasets
Download the datasets to `your_datapath`
1. [Totalsegmentator](https://github.com/wasserth/TotalSegmentator).
2. [DAP Atlas](https://github.com/alexanderjaus/AtlasDataset).

##### 1.3 Download the pre-trained checkpoint
We use Swin UNETR as an example.
```bash
cd direct_inference/pretrained_weights/
wget https://huggingface.co/MrGiovanni/SuPreM/resolve/main/supervised_suprem_swinunetr_2100.pth
cd ../../../
```

### 2. Inference & Evaluation
We show an example on TotalSegmentator dataset.
##### 2.1 Inference
```bash
cd tasks/TotalSegmentator/
datapath=/your_datapath/Totalsegmentator/
checkpoint_path=pretrained_weights/supervised_suprem_swinunetr_2100.pth
arch=swinunetr
python -W ignore inference.py --model_backbone $arch --log_name totalsegmentator.$arch --dataset_path $datapath --num_workers 8 --pretrain $checkpoint_path
```

##### 2.2 Evaluation
To accelerate the computation of the segmentation metrics, we employ a multi-process approach. The number of processes can be controlled by `num_worker`, which should be set according to the size of the memory.
```bash
datapath=/your_datapath/Totalsegmentator/
arch=swinunetr
python -W ignore evaluation.py --log_name totalsegmentator.$arch --num_workers 8 --dataset_path $datapath --pred_path out/totalsegmentator.$arch/pred/
```