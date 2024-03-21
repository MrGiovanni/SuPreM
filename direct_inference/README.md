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
- Prepare a text file (.txt extension) storing all the subfolder names(e.g., casename00001), one name per line.
  Note: this text file name will be assigned to the $datasetlist in step 3.

```
casename00001
casename00002
casename00003
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
backbone=unet # or swinunetr
pretrainpath=./pretrained_checkpoints/supervised_suprem_unet_2100.pth # or ./pretrained_weights/supervised_suprem_swinunetr_2100.pth
savepath=./inference
datasettxtpath=./dataset/dataset_list/
datasetlist=AbdomenAtlas1.0 # change to the txt file name, this txt file stores all the subfolder names (must be saved under datasettxtpath)
datarootpath=/scratch/zzhou82/data/AbdomenAtlas1.0Mini # change to /path/to/your/CT/scan/folders

cd SuPreM/direct_inference/
python -W ignore inference.py --save_dir $savepath --checkpoint $pretrainpath --dataset_list $datasetlist --data_root_path $datarootpath --data_txt_path $datasettxtpath --backbone $backbone --store_result
```
