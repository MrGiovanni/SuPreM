# Apply SuPreM to New CT Scans

##### 0. Prepare New CT Scans with Structured Folders
- Create a folder to hold all your CT scans
- Within this folder, create a separate subfolder for each CT scan (e.g., casename00001, casename00002).
- Place your new CT scan file in its corresponding subfolder and name the CT scan "ct.nii.gz".

```
/path/to/your/CT/scan/folders
    ├── casename00001
    │   └── ct.nii.gz
    ├── casename00002
    │   └── ct.nii.gz
    ├── casename00003
    │   └── ct.nii.gz
    ...
```
##### 1. Clone and setup the GitHub repository
```bash
git clone https://github.com/MrGiovanni/SuPreM
cd SuPreM/direct_inference/pretrained_checkpoints/
wget https://huggingface.co/MrGiovanni/SuPreM/resolve/main/supervised_suprem_swinunetr_2100.pth
wget https://huggingface.co/MrGiovanni/SuPreM/resolve/main/supervised_suprem_unet_2100.pth
```

##### 2 Create Environments
```bash
conda create -n suprem python=3.9
source activate suprem
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install monai[all]==0.9.0
cd SuPreM/
pip install -r requirements.txt
```

##### 3. Apply SuPreM to New CT Scans

```bash
datarootpath=/path/to/your/CT/scan/folders
# NEED MODIFICATION!!!

backbone=unet # or swinunetr
pretrainpath=./pretrained_checkpoints/supervised_suprem_unet_2100.pth # or ./pretrained_weights/supervised_suprem_swinunetr_2100.pth
savepath=./inference

cd SuPreM/direct_inference/
python -W ignore inference.py --save_dir $savepath.$backbone --checkpoint $pretrainpath --data_root_path $datarootpath --backbone $backbone --store_result --suprem
```
