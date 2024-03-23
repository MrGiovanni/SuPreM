# Let's Annotate *Vertebrae* :wink:

##### 0. Download CT scans

```bash
wget http://www.cs.jhu.edu/~zongwei/dataset/AbdomenAtlasDemo.tar.gz
tar -xzvf AbdomenAtlasDemo.tar.gz
```

The CT scans are organized in such a way:

```
AbdomenAtlasDemo
    ├── BDMAP_00000001
    │   └── ct.nii.gz
    ├── BDMAP_00000002
    │   └── ct.nii.gz
    ├── BDMAP_00000003
    │   └── ct.nii.gz
    ...
```

##### 1. Clone and setup the GitHub repository
```bash
git clone https://github.com/MrGiovanni/SuPreM
cd SuPreM/direct_inference/pretrained_checkpoints/
wget http://www.cs.jhu.edu/~zongwei/model/swin_unetr_totalsegmentator_vertebrae.pth
```

##### 2 Create environments
```bash
conda create -n suprem python=3.9
source activate suprem
cd SuPreM/
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install monai[all]==0.9.0
pip install -r requirements.txt
```

##### 3. Generate vertebrae masks by the AI

```bash
datarootpath=/path/to/your/AbdomenAtlasDemo
# NEED MODIFICATION!!!

pretrainpath=./pretrained_checkpoints/swin_unetr_totalsegmentator_vertebrae.pth
savepath=./AbdomenAtlasDemoPredict

cd SuPreM/direct_inference/
python -W ignore inference.py --save_dir $savepath --checkpoint $pretrainpath --data_root_path $datarootpath --customize
```

The vertebrae masks will be saved as
```
AbdomenAtlasDemoPredict
    ├── BDMAP_00000001
    │   ├── combined_labels.nii.gz
    │   └── segmentations
    │       ├── vertebrae_L5.nii.gz
    │       ├── vertebrae_L4.nii.gz
    │       ├── ...
    │       ├── vertebrae_L1.nii.gz
    │       ├── vertebrae_T12.nii.gz
    │       ├── vertebrae_T11.nii.gz
    │       ├── ...
    │       ├── vertebrae_T1.nii.gz
    │       ├── vertebrae_C7.nii.gz
    │       ├── vertebrae_C6.nii.gz
    │       ├── ...
    │       └── vertebrae_C1.nii.gz
    ├── BDMAP_00000002
    │   ├── combined_labels.nii.gz
    │   └── segmentations
    │       ├── vertebrae_L5.nii.gz
    │       ├── vertebrae_L4.nii.gz
    │       ├── ...
    │       ├── vertebrae_L1.nii.gz
    │       ├── vertebrae_T12.nii.gz
    │       ├── vertebrae_T11.nii.gz
    │       ├── ...
    │       ├── vertebrae_T1.nii.gz
    │       ├── vertebrae_C7.nii.gz
    │       ├── vertebrae_C6.nii.gz
    │       ├── ...
    │       └── vertebrae_C1.nii.gz
    ├── BDMAP_00000003
    │   ├── combined_labels.nii.gz
    │   └── segmentations
    │       ├── vertebrae_L5.nii.gz
    │       ├── vertebrae_L4.nii.gz
    │       ├── ...
    │       ├── vertebrae_L1.nii.gz
    │       ├── vertebrae_T12.nii.gz
    │       ├── vertebrae_T11.nii.gz
    │       ├── ...
    │       ├── vertebrae_T1.nii.gz
    │       ├── vertebrae_C7.nii.gz
    │       ├── vertebrae_C6.nii.gz
    │       ├── ...
    │       └── vertebrae_C1.nii.gz
    ...
```

##### 4. [Important!] Postprocess vertebrae masks

Please check the AI-predicted vertebrae masks and the original CT scans. To ease the visualization, you can plot videos by modifying our [plot_video_multiprocessing.py](https://github.com/MrGiovanni/SuPreM/blob/main/utils/plot_video_multiprocessing.py). Based on the medical knowledge below, design an automatic postprocessing to refine the vertebrae masks. The postprocessing should be formatted in a separated python file `postprocessing_vertebrae.py`.

![Vertebral anatomy](https://i0.wp.com/aneskey.com/wp-content/uploads/2023/08/f01-01-9780323882262.jpg)
</div>

