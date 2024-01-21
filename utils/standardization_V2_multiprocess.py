'''
python standardization_V2_multiprocess.py --ORIGINAL_ROOT_DIR /Users/zongwei.zhou/Desktop/BodyMapsSmall --REVISED_ROOT_DIR /Users/zongwei.zhou/Desktop/BodyMapsSmallCore
'''

import os
import nibabel as nib
import numpy as np
import shutil
import argparse
import glob
import copy
from tqdm import tqdm
from skimage.measure import label
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

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

def rename_id(name):
    name = name.replace('01_Multi-Atlas_Labeling', '01_BTCV')
    name = name.replace('18_FLARE23_FLARE23_', '18_FLARE23_')
    
    return name

def standardize_and_save(CT_ID, original_root_dir, revised_root_dir):
    original_ct_file = os.path.join(original_root_dir, CT_ID, 'ct.nii.gz')
    revised_ct_file = os.path.join(revised_root_dir, CT_ID, 'ct.nii.gz')
    revised_ct_file = rename_id(revised_ct_file)
    
    original_mask_files = glob.glob(os.path.join(original_root_dir, CT_ID, 'segmentations/*.nii.gz'))
    revised_mask_files = [s.replace(original_root_dir, revised_root_dir) for s in original_mask_files]
    revised_mask_files = [rename_id(s) for s in revised_mask_files]

    # Ensure the revised directories exist
    os.makedirs(os.path.dirname(revised_ct_file), exist_ok=True)
    for mask_file in revised_mask_files:
        os.makedirs(os.path.dirname(mask_file), exist_ok=True)

    # Perform standardization and saving
    try:
        standardization(original_ct_file=original_ct_file, revised_ct_file=revised_ct_file,
                        original_mask_file=original_mask_files, revised_mask_file=revised_mask_files,
                        mask_type=np.int8)
    except Exception as e:
        print(f"Error processing {CT_ID}: {e}")
    

def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--ORIGINAL_ROOT_DIR', default=None, type=str, help='original root directory')
    parser.add_argument('--REVISED_ROOT_DIR', default=None, type=str, help='revised root directory')
    
    args = parser.parse_args()

    assert args.ORIGINAL_ROOT_DIR is not None
    assert args.REVISED_ROOT_DIR is not None

    CT_ID_LIST = glob.glob(os.path.join(args.ORIGINAL_ROOT_DIR, '*'))
    CT_ID_LIST = [os.path.basename(s) for s in CT_ID_LIST]
    
    print('>> {} CPU cores are secured.'.format(cpu_count()))
    
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        futures = {executor.submit(standardize_and_save, CT_ID, args.ORIGINAL_ROOT_DIR, args.REVISED_ROOT_DIR): CT_ID
                   for CT_ID in CT_ID_LIST}
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            CT_ID = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {CT_ID}: {e}")

if __name__ == "__main__":
    main()
