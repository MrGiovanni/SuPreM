'''
python crop_jhh_dataset.py --ctpath /Users/zongwei.zhou/Desktop/14_FELIX/img --maskpath /Users/zongwei.zhou/Desktop/14_FELIX/label_6cls --savepath /Users/zongwei.zhou/Desktop/JHH_ROI_0.5mm
'''

import os
import numpy as np
import nibabel as nib
import cc3d
import argparse
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

CLASSNAME = ['pancreas',
             'pdac',
             'cyst',
             'pnet',
            ]

def intensity_clip(ct):
    ct[ct < -1000] = -1000
    ct[ct > 1000] = 1000
    
    return ct

def process_mask_and_ct(patientID, args):
        
    mask_path = os.path.join(args.maskpath, patientID + '.nii.gz')
    ct_path = os.path.join(args.ctpath, patientID + '.nii.gz')

    mask_nifti = nib.load(mask_path)
    mask = mask_nifti.get_fdata()

    ct_nifti = nib.load(ct_path)
    ct = ct_nifti.get_fdata()

    # Create a union of labels 1 through 5
    union_mask = np.isin(mask, [1, 2, 3, 4, 5])

    # Get a labeling of the k largest objects in the image.
    # The output will be relabeled from 1 to N.
    labels_out, _ = cc3d.largest_k(
        union_mask, k=1, 
        connectivity=26, delta=0,
        return_N=True,
        )
    union_mask *= (labels_out > 0) # to get original labels
    mask[union_mask == 0] = 0

    slice_x = np.any(union_mask, axis=(1, 2))
    slice_y = np.any(union_mask, axis=(0, 2))
    slice_z = np.any(union_mask, axis=(0, 1))

    x_start = max(0, np.argmax(slice_x) - args.x_padding)
    x_end = min(len(slice_x), len(slice_x) - np.argmax(slice_x[::-1]) + args.x_padding)

    y_start = max(0, np.argmax(slice_y) - args.y_padding)
    y_end = min(len(slice_y), len(slice_y) - np.argmax(slice_y[::-1]) + args.y_padding)

    z_start = max(0, np.argmax(slice_z) - args.z_padding)
    z_end = min(len(slice_z), len(slice_z) - np.argmax(slice_z[::-1]) + args.z_padding)

    cropped_mask = mask[x_start:x_end, y_start:y_end, z_start:z_end]
    cropped_ct = intensity_clip(ct[x_start:x_end, y_start:y_end, z_start:z_end])
    
    if args.saveformat == 'bodymaps':

        if not os.path.exists(os.path.join(args.savepath, patientID)):
            os.makedirs(os.path.join(args.savepath, patientID))
        cropped_ct_nifti = nib.Nifti1Image(cropped_ct.astype(np.int16), ct_nifti.affine, ct_nifti.header)
        nib.save(cropped_ct_nifti, os.path.join(args.savepath, patientID, 'ct.nii.gz'))

        combined_labels = np.where(np.isin(cropped_mask, [1, 2]), 1, cropped_mask)
        combined_labels_nifti = nib.Nifti1Image(combined_labels.astype(np.int8), mask_nifti.affine, mask_nifti.header)
        nib.save(combined_labels_nifti, os.path.join(args.savepath, patientID, 'combined_labels.nii.gz'))

        if not os.path.exists(os.path.join(args.savepath, patientID, 'segmentations')):
            os.makedirs(os.path.join(args.savepath, patientID, 'segmentations'))
        
        for class_name in CLASSNAME:
            if class_name == 'pancreas':
                cls_mask = np.where(np.isin(cropped_mask, [1, 2, 3, 4, 5]), 1, cropped_mask)
            elif class_name == 'pdac':
                cls_mask = (cropped_mask == 3)
            elif class_name == 'cyst':
                cls_mask = (cropped_mask == 4)
            elif class_name == 'pnet':
                cls_mask = (cropped_mask == 5)
                
            cropped_mask_nifti = nib.Nifti1Image(cls_mask.astype(np.int8), mask_nifti.affine, mask_nifti.header)
            nib.save(cropped_mask_nifti, os.path.join(args.savepath, patientID, 'segmentations', class_name + '.nii.gz'))

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--ctpath', default=None, type=str, help='the directory of CT volumes')
    parser.add_argument('--maskpath', default=None, type=str, help='the directory of masks')
    parser.add_argument('--savepath', default=None, type=str, help='the directory of masks')
    parser.add_argument('--saveformat', default='bodymaps', type=str, help='bodymaps')
    parser.add_argument('--x_padding', default=48, type=int, help='x padding')
    parser.add_argument('--y_padding', default=48, type=int, help='y padding')
    parser.add_argument('--z_padding', default=48, type=int, help='z padding')
    
    args = parser.parse_args()

    assert args.ctpath is not None
    assert args.maskpath is not None
    assert args.savepath is not None

    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)
    
    patientIDlist = [f.split('.nii')[0] for f in os.listdir(args.maskpath) if os.path.isfile(os.path.join(args.maskpath, f))]
    
    # for patientID in tqdm(patientIDlist, desc="Processing masks"):
    #     process_mask_and_ct(patientID, args)
    
    print('>> {} CPU cores are secured.'.format(cpu_count()))

    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        futures = {executor.submit(process_mask_and_ct, patientID, args): patientID for patientID in patientIDlist}

        for future in tqdm(as_completed(futures), total=len(futures)):
            patientID = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {patientID}: {e}")

if __name__ == "__main__":
    main()
