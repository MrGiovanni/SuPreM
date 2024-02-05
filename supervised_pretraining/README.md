# Pre-training SuPreM on the AbdomenAtlas 1.1 dataset

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

##### Next step: TBA