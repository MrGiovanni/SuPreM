'''
python standardization_V2.py --ORIGINAL_ROOT_DIR /data/zzhou82/data/AbdomenAtlas --REVISED_ROOT_DIR /data/zzhou82/data/AbdomenAtlas_Core --start 0 --end 100
'''

import os
import nibabel as nib
import numpy as np
import shutil
import argparse
import glob
from tqdm import tqdm
from skimage.measure import label

import warnings
warnings.filterwarnings("ignore")

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def find_largest_subarray_bounds(arr, low_threshold, high_threshold):
    
    # Find the indices where the condition is True
    condition = (arr > low_threshold) & (arr < high_threshold)
    condition = getLargestCC(condition)
    x, y, z = np.where(condition)

    if not len(x):
        return (0,0,0), (0,0,0)  # No values above threshold

    # Find min and max indices along each dimension
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    min_z, max_z = np.min(z), np.max(z)

    return (min_x, min_y, min_z), (max_x, max_y, max_z)

def crop_largest_subarray(arr, low_threshold, high_threshold, case_name=None):
    
    (min_x, min_y, min_z), (max_x, max_y, max_z) = find_largest_subarray_bounds(arr, low_threshold, high_threshold)
    if max_x - min_x < 50 or max_y - min_y < 50 or max_z - min_z < 5:
        print('ERROR in {}'.format(case_name))
    
    return (min_x, min_y, min_z), (max_x, max_y, max_z)

def standardization(original_ct_file, revised_ct_file, 
                    original_mask_file=None, revised_mask_file=None,
                    image_type=np.int16, mask_type=np.int8,
                   ):
    
    img = nib.load(original_ct_file)
    data = np.array(img.dataobj)

    data[data > 1000] = 1000
    data[data < -1000] = -1000
    
    (min_x, min_y, min_z), (max_x, max_y, max_z) = crop_largest_subarray(arr=data, 
                                                                         low_threshold=-100, 
                                                                         high_threshold=100, 
                                                                         case_name=original_ct_file.split('/')[-2])
    data = data[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]

    data = nib.Nifti1Image(data, img.affine, img.header)
    data.set_data_dtype(image_type)
    data.get_data_dtype(finalize=True)
    
    nib.save(data, revised_ct_file)
    if original_mask_file is not None and revised_mask_file is not None:
        
        for original, revised in zip(original_mask_file, revised_mask_file):
            img = nib.load(original)
            data = np.array(img.dataobj)
            mask = data[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]

            mask = nib.Nifti1Image(mask, img.affine, img.header)
            mask.set_data_dtype(mask_type)
            mask.get_data_dtype(finalize=True)

            nib.save(mask, revised)

def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--start', default=0, type=int, help='start data index')
    parser.add_argument('--end', default=10, type=int, help='end data index')
    parser.add_argument('--ORIGINAL_ROOT_DIR', default=None, type=str, help='original root directory')
    parser.add_argument('--REVISED_ROOT_DIR', default=None, type=str, help='revised root directory')
    
    args = parser.parse_args()

    assert args.ORIGINAL_ROOT_DIR is not None
    assert args.REVISED_ROOT_DIR is not None
    
    CT_ID_LIST = glob.glob(os.path.join(args.ORIGINAL_ROOT_DIR, '*'))
    CT_ID_LIST = [s.split('/')[-1] for s in CT_ID_LIST]
    
    assert args.end <= len(CT_ID_LIST)
    assert args.start <= len(CT_ID_LIST)

    for CT_ID in tqdm(CT_ID_LIST[args.start:args.end]):
        ORIGINAL_CT_FILE = os.path.join(args.ORIGINAL_ROOT_DIR, CT_ID, 'ct.nii.gz')
        if not os.path.exists(os.path.join(args.REVISED_ROOT_DIR, CT_ID)):
            os.makedirs(os.path.join(args.REVISED_ROOT_DIR, CT_ID))
        REVISED_CT_FILE = os.path.join(args.REVISED_ROOT_DIR, CT_ID, 'ct.nii.gz')

        ORIGINAL_MASK_FILE = glob.glob(os.path.join(args.ORIGINAL_ROOT_DIR, CT_ID, 'segmentations/*.nii.gz'))
        if not os.path.exists(os.path.join(args.REVISED_ROOT_DIR, CT_ID, 'segmentations')):
            os.makedirs(os.path.join(args.REVISED_ROOT_DIR, CT_ID, 'segmentations'))
        REVISED_MASK_FILE = [s.replace(args.ORIGINAL_ROOT_DIR, args.REVISED_ROOT_DIR) for s in ORIGINAL_MASK_FILE]
        standardization(original_ct_file=ORIGINAL_CT_FILE, revised_ct_file=REVISED_CT_FILE,
                        original_mask_file=ORIGINAL_MASK_FILE, revised_mask_file=REVISED_MASK_FILE,
                        mask_type=np.int8,
                       )

if __name__ == "__main__":
    main()
