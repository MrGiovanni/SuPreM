# Predict organ masks on unseen CT volumes using SuPreM (direct inference)

##### 0. Create a virtual environment (optional)

```bash
conda create -n suprem python=3.8
source activate suprem
```

##### 1. Clone the GitHub repository

```bash
git clone https://github.com/MrGiovanni/SuPreM
cd SuPreM/
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install monai[all]==0.9.0
pip install -r requirements.txt
```

##### 2. Download the pre-trained Swin UNETR checkpoint

```bash
cd target_applications/totalsegmentator/pretrained_weights/
wget https://huggingface.co/MrGiovanni/SuPreM/resolve/main/supervised_suprem_swinunetr_2100.pth
cd ../../../
```

##### Next step: TBA (using TotalSegmentator as an example)