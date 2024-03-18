# Apply SuPreM to New CT Scans


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
datasetlist=AbdomenAtlas1.0 # change to the txt file name, this txt file stores all the names of new CT scans (must be saved under datasettxtpath)
datarootpath=/scratch/zzhou82/data/AbdomenAtlas1.0Mini

cd SuPreM/direct_inference/
python -W ignore inference.py --save_dir $savepath --resume $pretrainpath --dataset_list $datasetlist --data_root_path $datarootpath --data_txt_path $datasettxtpath --backbone $backbone --store_result
```
