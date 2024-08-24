<h1 align="center">SuPreM</h1>
<h3 align="center" style="font-size: 20px; margin-bottom: 4px">For Pancreatic Tumor Detection</h3>
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


#### STEP 0. Setup


###### STEP 0.1 Create a virtual environment (optional)

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

###### STEP 0.2 Clone the GitHub repository

```bash
git clone https://github.com/MrGiovanni/SuPreM
cd SuPreM/
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install monai[all]==0.9.0
pip install pip==23.3.1
pip install -r requirements.txt
```

###### STEP 0.3 Download the pre-trained checkpoints

```bash
cd target_applications/pancreas_tumor_detection/pretrained_weights/
wget https://huggingface.co/MrGiovanni/SuPreM/resolve/main/supervised_suprem_swinunetr_2100.pth
wget https://huggingface.co/MrGiovanni/SuPreM/resolve/main/supervised_suprem_segresnet_2100.pth
wget https://huggingface.co/MrGiovanni/SuPreM/resolve/main/supervised_suprem_unet_2100.pth
cd ../
```

#### STEP 1. Train - train an AI on the training set

```bash
# Multiple GPUs
bash shell_scripts/step1.train.multigpu.sh segresnet pretrained_weights/supervised_suprem_segresnet_2100.pth 8 4 >> logs/segresnet.tiny.jhh.txt
# bash shell_scripts/step1.train.multigpu.sh swinunetr pretrained_weights/supervised_suprem_swinunetr_2100.pth  2 2 >> logs/swinunetr.jhh.txt
```

```bash
# Single GPU
bash shell_scripts/step1.train.singlegpu.sh segresnet pretrained_weights/supervised_suprem_segresnet_2100.pth 8 4 >> logs/segresnet.tiny.jhh.txt
# bash shell_scripts/step1.train.singlegpu.sh swinunetr pretrained_weights/supervised_suprem_swinunetr_2100.pth  2 2 >> logs/swinunetr.jhh.txt
```

#### STEP 2. Test - test the AI on the test set

###### STEP 2.1 Make predictions

```bash
# Multiple GPUs
bash shell_scripts/step2.inference.multigpu.sh segresnet out/segresnet.jhh/model.pth segresnet.jhh 8
# bash shell_scripts/step2.inference.multigpu.sh swinunetr out/swinunetr.jhh/model.pth swinunetr.jhh 8
```

```bash
# Single GPU
bash shell_scripts/step2.inference.singlegpu.sh segresnet out/segresnet.jhh/model.pth segresnet.jhh
```

###### If you wanted to test multiple AI checkpoints

```bash
for epoch in {270..100..-10}; do bash shell_scripts/step2.inference.multigpu.sh segresnet out/segresnet.jhh/checkpoint_epoch_$epoch.pth segresnet.jhh.checkpoint.$epoch 8; done
```

###### STEP 2.2 Perform post-processing, report eval metrics (e.g., sensitivity, specificity, & PPV), save TP, TN, FP, FN IDs and visuals in the error_analysis folder

```bash
bash shell_scripts/step3.eval.sh inference/segresnet.jhh /data/zzhou82/data/JHH_ROI_0.5mm segresnet.jhh >> logs/segresnet.jhh.txt
```

###### If you wanted to evaluate multiple AI predictions
```bash
for epoch in {270..100..-10}; do bash shell_scripts/step3.eval.sh inference/segresnet.jhh.checkpoint.$epoch /data/zzhou82/data/JHH_ROI_0.5mm segresnet.jhh.checkpoint.$epoch >> logs/segresnet.jhh.checkpoint.$epoch.txt; done 
```

#### Results

<p align="center"><img width="100%" src="document/roc_curve.png" /></p>

<p align="center"><img width="100%" src="document/roc_curve_zoomin.png" /></p>
