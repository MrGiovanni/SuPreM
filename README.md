<div align="center">
 
![logo](document/fig_suprem_logo.png)  
**Subscribe us: https://groups.google.com/u/2/g/bodymaps**  
</div>

We developed a suite of pre-trained 3D models, named **SuPreM**, that combined the best of large-scale datasets and per-voxel annotations, showing the transferability across a range of 3D medical imaging tasks.

## Paper

<b>How Well Do Supervised 3D Models Transfer to Medical Imaging Tasks?</b> <br/>
[Wenxuan Li](https://scholar.google.com/citations?hl=en&user=tpNZM2YAAAAJ), [Alan Yuille](https://www.cs.jhu.edu/~ayuille/), and [Zongwei Zhou](https://www.zongweiz.com/)<sup>*</sup> <br/>
Johns Hopkins University  <br/>
International Conference on Learning Representations (ICLR) 2024 (oral; top 1.2%) <br/>
[paper](https://www.cs.jhu.edu/~alanlab/Pubs23/li2023suprem.pdf) | [code](https://github.com/MrGiovanni/SuPreM) | slides | talk

<b>Transitioning to Fully-Supervised Pre-Training with Large-Scale Radiology ImageNet for Improved AI Transferability in Three-Dimensional Medical Segmentation</b> <br/>
[Wenxuan Li](https://scholar.google.com/citations?hl=en&user=tpNZM2YAAAAJ)<sup>1</sup>, [Junfei Xiao](https://lambert-x.github.io/)<sup>1</sup>, [Jie Liu](https://ljwztc.github.io/)<sup>2</sup>, [Yucheng Tang](https://scholar.google.com/citations?hl=en&user=0xheliUAAAAJ)<sup>3</sup>, [Alan Yuille](https://www.cs.jhu.edu/~ayuille/)<sup>1</sup>, and [Zongwei Zhou](https://www.zongweiz.com/)<sup>1,*</sup> <br/>
<sup>1</sup>Johns Hopkins University  <br/>
<sup>2</sup>City University of Hong Kong  <br/>
<sup>3</sup>NVIDIA  <br/>
Radiological Society of North America (RSNA) 2023  <br/>
[abstract](document/rsna_abstract.pdf) | [code](https://github.com/MrGiovanni/SuPreM) | [slides](document/rsna_slides.pdf) | talk

**&#9733; We have maintained a document for [Frequently Asked Questions](document/frequently_asked_questions.md).**

**&#9733; We have provided a list of publications about 3D medical pre-training in [Awesome Medical Pre-Training](document/awesome_medical_pretraining.md) [![Awesome](https://awesome.re/badge.svg)](https://awesome.re).**

## An Extensive Dataset: AbdomenAtlas 1.1

The release of AbdomenAtlas 1.0 can be found at https://github.com/MrGiovanni/AbdomenAtlas

AbdomenAtlas 1.1 is an extensive dataset of 9,262 CT volumes with per-voxel annotation of **25 organs** and pseudo annotations for **seven types of tumors**, enabling us to *finally* perform supervised pre-training of AI models at scale. Based on AbdomenAtlas 1.1, we also provide a suite of pre-trained models comprising several widely recognized AI models. 

<p align="center"><img width="100%" src="document/fig_benchmark.png" /></p>

Prelimianry benchmark showed that supervised pre-training strikes as a preferred choice in terms of performance and efficiency compared with self-supervised pre-training. 

We anticipate that the release of large, annotated datasets (AbdomenAtlas 1.1) and the suite of pre-trained models (SuPreM) will bolster collaborative endeavors in establishing Foundation Datasets and Foundation Models for the broader applications of 3D volumetric medical image analysis.

## A Suite of Pre-trained Models: SuPreM

The following is a list of supported model backbones in our collection. Select the appropriate family of backbones and click to expand the table, download a specific backbone and its pre-trained weights (`name` and `download`), and save the weights into `./pretrained_weights/`. More backbones will be added along time. **Please suggest the backbone in [this channel](https://github.com/MrGiovanni/SuPreM/issues/1) if you want us to pre-train it on AbdomenAtlas 1.1 containing 9,262 annotated CT volumes.**

<details>
<summary style="margin-left: 25px;">Swin UNETR</summary>
<div style="margin-left: 25px;">

| name | params | pre-trained data | resources | download |
|:----  |:----  |:----  |:----  |:----  |
| [Tang et al.](https://openaccess.thecvf.com/content/CVPR2022/papers/Tang_Self-Supervised_Pre-Training_of_Swin_Transformers_for_3D_Medical_Image_Analysis_CVPR_2022_paper.pdf) | 62.19M | 5050 CT | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/research-contributions.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/research-contributions) | [weights](https://www.dropbox.com/scl/fi/7v0ohio4dhaod8azg0unw/self_supervised_nv_swin_unetr_5050.pt?rlkey=rq03hg6guq9kpi4uqpbkte41d&dl=0) |
| [Jose Valanaras et al.](https://arxiv.org/pdf/2307.16896.pdf](https://arxiv.org/pdf/2307.16896.pdf)) | 62.19M | 50000 CT/MRI | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/research-contributions.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/research-contributions) | [weights](https://www.dropbox.com/scl/fi/ymw3zt4aawa1fvo1o7saj/self_supervised_nv_swin_unetr_50000.pth?rlkey=sanmzwl9ez1xub9y94fiozogk&dl=0) |
| [Universal Model](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_CLIP-Driven_Universal_Model_for_Organ_Segmentation_and_Tumor_Detection_ICCV_2023_paper.pdf) | 62.19M | 2100 CT | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) | [weights](https://www.dropbox.com/scl/fi/v7fg5cesxx6hra6stln09/supervised_clip_driven_universal_swin_unetr_2100.pth?rlkey=2pubjk679ai5s59g8884318r8&dl=0) |
| SuPreM | 62.19M | 2100 CT | ours :star2: | [weights](https://www.dropbox.com/scl/fi/gd1d7k9mac5azpwurds66/supervised_suprem_swinunetr_2100.pth?rlkey=xoqr7ey52rnese2k4hwmrlqrt&dl=0) |

</div>
</details>

<details>
<summary style="margin-left: 25px;">U-Net</summary>
<div style="margin-left: 25px;">

| name | params | pre-trained data | resources | download |
|:----  |:----  |:----  |:----  |:----  |
| [Models Genesis](http://www.cs.toronto.edu/~liang/Publications/ModelsGenesis/MICCAI_2019_Full.pdf) | 19.08M | 623 CT | [![GitHub stars](https://img.shields.io/github/stars/MrGiovanni/ModelsGenesis.svg?logo=github&label=Stars)](https://github.com/MrGiovanni/ModelsGenesis) | [weights](https://www.dropbox.com/scl/fi/h9006v3bhxjlg6x1yvcge/self_supervised_models_genesis_unet_620.pt?rlkey=2wqm80i5hvrktpk6n7ye3hc55&dl=0) |
| [UniMiSS](https://link.springer.com/chapter/10.1007/978-3-031-19803-8_33) | tiny | 5022 CT&MRI | [![GitHub stars](https://img.shields.io/github/stars/YtongXie/UniMiSS-code.svg?logo=github&label=Stars)](https://github.com/YtongXie/UniMiSS-code) | [weights](https://www.dropbox.com/scl/fi/joxv6gwhizcg5pmgxtoy7/self_supervised_unimiss_nnunet_tiny_5022.pth?rlkey=kj2wzl6t5fxfkxuwl18r1jfer&dl=0) |
|  | small |  |  | [weights](https://www.dropbox.com/scl/fi/mej5qfc5w6dmtsgn7hsbs/self_supervised_unimiss_nnunet_small_5022.pth?rlkey=k0hzktopzh8ih20aou241r9rb&dl=0) |
| [Med3D](https://arxiv.org/pdf/1904.00625.pdf) | 85.75M | 1638 CT | [![GitHub stars](https://img.shields.io/github/stars/Tencent/MedicalNet.svg?logo=github&label=Stars)](https://github.com/Tencent/MedicalNet) | [weights](https://www.dropbox.com/scl/fi/dv2u26f5ibwvkd0wgf8sj/supervised_med3D_residual_unet_1623.pth?rlkey=h7ddvgilv27esypjh8w4hrhgb&dl=0) |
| [DoDNet](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_DoDNet_Learning_To_Segment_Multi-Organ_and_Tumors_From_Multiple_Partially_CVPR_2021_paper.pdf) | 17.29M | 920 CT | [![GitHub stars](https://img.shields.io/github/stars/jianpengz/DoDNet.svg?logo=github&label=Stars)](https://github.com/jianpengz/DoDNet) | [weights](https://www.dropbox.com/scl/fi/wf5xuoqe2pa8xky6rwhv1/supervised_dodnet_unet_920.pth?rlkey=c6g94jxukuan4osxldbrpwbix&dl=0) |
| [Universal Model](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_CLIP-Driven_Universal_Model_for_Organ_Segmentation_and_Tumor_Detection_ICCV_2023_paper.pdf) | 19.08M | 2100 CT | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) | [weights](https://www.dropbox.com/scl/fi/k6yffxaavofwtkut076t0/supervised_clip_driven_universal_unet_2100.pth?rlkey=myf71pu6v8j16bwv4ouoz4xov&dl=0) |
| SuPreM | 19.08M | 2100 CT | ours :star2: | [weights](https://www.dropbox.com/scl/fi/r5ti0vj6sgrengvvmez1k/supervised_suprem_unet_2100.pth?rlkey=i4rw4xw44vpjpz75eq32fk0rb&dl=0) |

</div>
</details>

<details>
<summary style="margin-left: 25px;">SegResNet</summary>
<div style="margin-left: 25px;">

| name | params | pre-trained data | resources | download |
|:----  |:----  |:----  |:----  |:----  |
| SuPreM | 470.13M | 2100 CT | ours :star2: | [weights](https://www.dropbox.com/scl/fi/hq10omgcie7mdsxo34bjg/supervised_suprem_segresnet_2100.pth?rlkey=pwwes2dnlo2t6na80kioljbmm&dl=0) |

</div>
</details>

## Acknowledgement

This work was supported by the Lustgarten Foundation for Pancreatic Cancer Research and the McGovern Foundation. The segmentation backbone is based on [Swin UNETR](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_btcv_segmentation_3d.ipynb); we appreciate the effort of the [MONAI Team](https://monai.io/) to provide and maintain open-source code to the community. Paper content is covered by patents pending.
