import os
import numpy as np
import SimpleITK as sitk
import shutil
import nibabel as nib
from tqdm import tqdm
import time

os.environ['MKL_THREADING_LAYER'] = 'GNU'
#all you need to modify
gpu_id = 0
output_path = "AbdomenAtlas/Prediction/"
test_data_path = "AbdomenAtlas/TestExamples/"
preprocessed_data_path = "AbdomenAtlas/Preprocessed_data/"

os.makedirs(os.path.join(preprocessed_data_path, "image"),exist_ok=True)
os.makedirs(output_path,exist_ok=True)

#inference
def NiiDataRead(path, as_type=np.float32):
    img = sitk.ReadImage(path)
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    direction = img.GetDirection()
    img_it = sitk.GetArrayFromImage(img).astype(as_type)
    return img_it, spacing, origin, direction

def NiiDataWrite(path, prediction_final, spacing, origin, direction):
    img = sitk.GetImageFromArray(prediction_final, isVector=False)
    # print(img.GetSize())
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    img.SetDirection(direction)
    sitk.WriteImage(img, path)

def process_nii(file_path, save_path):
    nii = nib.load(file_path)
    img = nii.get_fdata()
    header = nii.header

    spacing = header.get_zooms()
    if spacing[0] == spacing[1] == spacing[2]:
        return
    if spacing[0] == spacing[1]:
        return
    elif spacing[0] == spacing[2]:
        reordered_img = img.transpose(0, 2, 1)
        reordered_spacing = (spacing[0], spacing[2], spacing[1])
    elif spacing[1] == spacing[2]:
        reordered_img = img.transpose(1, 2, 0)
        reordered_spacing = (spacing[1], spacing[2], spacing[0])
    else:
        return
    affine = np.diag(reordered_spacing + (1,))
    new_nii = nib.Nifti1Image(reordered_img, affine=affine)
    new_header = new_nii.header.copy()
    new_header.set_zooms(reordered_spacing)
    new_spacing = new_header.get_zooms()
    print("transpose: ", spacing, "-->", new_spacing, "Careful Evaluation!")
    nib.save(nib.Nifti1Image(reordered_img, affine=new_nii.affine, header=new_header), save_path)
    
def processing(sample_name, img):

    os.makedirs(os.path.join(preprocessed_data_path, "image", sample_name),exist_ok=True)
    shutil.copy(img, os.path.join(preprocessed_data_path, "image", sample_name, sample_name + "_0000.nii.gz"))
    process_nii(os.path.join(preprocessed_data_path, "image", sample_name, sample_name + "_0000.nii.gz"), os.path.join(preprocessed_data_path, "image", sample_name, sample_name + "_0000.nii.gz"))


def deprocessing(sample_name):
    os.makedirs(os.path.join(output_path, sample_name, "predictions"),exist_ok=True)
    seg_total, spacing, origin, direction  = NiiDataRead(os.path.join(output_path, sample_name, sample_name+".nii.gz"))

    for itr, class_name in enumerate(eval_class):
        tmp_seg = np.zeros_like(seg_total)
        tmp_seg[seg_total == (itr+1)] = 1
        NiiDataWrite(os.path.join(output_path, sample_name, "predictions", class_name+".nii.gz"), tmp_seg, spacing, origin, direction)


eval_class = ["aorta", "gall_bladder", "kidney_left", "kidney_right", "liver", "pancreas", "postcava", "spleen", "stomach"]
all_data_name = os.listdir(test_data_path)
all_data_name  = [name for name in all_data_name if not name.endswith(".DS_Store")]
for data_name in tqdm(all_data_name):
    sample_name = data_name
    img = os.path.join(test_data_path, data_name, "ct.nii.gz")
    processing(sample_name, img)
    os.makedirs(os.path.join(output_path, sample_name),exist_ok=True)
    os.system(f'CUDA_VISIBLE_DEVICES={gpu_id} python predict_simple.py -i {os.path.join(preprocessed_data_path, "image", sample_name)} \
            -mp checkpoints/ -o {os.path.join(output_path, sample_name)} -p nnUNetPlansv2.1 -t Task066_AbdomenAtlas1.0 -m 3d_fullres  -tr nnUNetTrainerV2_ResTrans -f all -chk model_best')
    deprocessing(sample_name)


