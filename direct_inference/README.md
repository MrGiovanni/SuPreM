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

### Option 1: 

[HuggingFace 🤗](https://huggingface.co/qicq1c/SuPreM)
```bash
inputs_data=/path/to/your/CT/scan/folders
outputs_data=/path/to/your/output/folders

# If using singularity
wget https://huggingface.co/qicq1c/SuPreM/resolve/main/suprem_final.sif
SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 singularity run --nv -B $inputs_data:/workspace/inputs -B $outputs_data:/workspace/outputs suprem_final.sif

# If using docker
docker pull qchen99/suprem:v1
sudo docker container run --gpus "device=0" -m 128G --rm -v $inputs_data:/workspace/inputs/ -v $outputs_data:/workspace/outputs/ qchen99/suprem:v1 /bin/bash -c "sh predict.sh"
```

### Option 2:

##### 1. Clone the GitHub repository
```bash
git clone https://github.com/MrGiovanni/SuPreM
cd SuPreM/direct_inference/pretrained_checkpoints/
wget https://huggingface.co/MrGiovanni/SuPreM/resolve/main/supervised_suprem_swinunetr_2100.pth
wget https://huggingface.co/MrGiovanni/SuPreM/resolve/main/supervised_suprem_unet_2100.pth
```

##### 2 Create environments
```bash
conda create -n suprem python=3.9
source activate suprem
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install monai[all]==0.9.0
cd SuPreM/
pip install -r requirements.txt
```

##### 3. Apply SuPreM to new CT scans

```bash
datarootpath=/path/to/your/CT/scan/folders
# NEED MODIFICATION!!!

backbone=unet # or swinunetr
pretrainpath=./pretrained_checkpoints/supervised_suprem_unet_2100.pth # or ./pretrained_weights/supervised_suprem_swinunetr_2100.pth
savepath=./inference

cd SuPreM/direct_inference/
python -W ignore inference.py --save_dir $savepath.$backbone --checkpoint $pretrainpath --data_root_path $datarootpath --backbone $backbone --store_result --suprem
```
