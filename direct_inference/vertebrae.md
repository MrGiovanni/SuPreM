<h1 align="center">SuPreM</h1>
<h3 align="center" style="font-size: 20px; margin-bottom: 4px">Apply to Vertebrae Segmentation</h3>
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
cd ..
```

<details>
<summary style="margin-left: 25px;">[Option] if you get certificate issues when using wget</summary>
<div style="margin-left: 25px;">

```bash
wget --no-check-certificate http://www.cs.jhu.edu/~zongwei/model/swin_unetr_totalsegmentator_vertebrae.pth
```

</div>
</details>


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
datarootpath=/path/to/your/AbdomenAtlasDemo # NEED MODIFICATION!!!

pretrainpath=./pretrained_checkpoints/swin_unetr_totalsegmentator_vertebrae.pth
savepath=./AbdomenAtlasDemoPredict

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

Please check the AI-predicted vertebrae masks and the original CT scans. If you look closely at the AI-predicted masks, you will see many errors. Please design an automatic postprocessing to reduce these errors as many as you can. The postprocessing should be formatted in a separated python file `postprocessing_vertebrae.py`.

This is an illustration of vertebrae (and rib) label refinement.

![Refinement](https://github.com/MrGiovanni/SuPreM/blob/main/document/LetsSegmentVertebrae.png)
</div>

To identify the errors, you will need some knowledge about vertebrae in the human body as follow.

![Vertebral anatomy](https://i0.wp.com/aneskey.com/wp-content/uploads/2023/08/f01-01-9780323882262.jpg)
</div>
