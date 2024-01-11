'''
python standardization_V3.py
'''

import os
import nibabel as nib
import numpy as np
import shutil
import glob
import h5py
from tqdm.notebook import tqdm
from skimage.measure import label

def nifti_to_h5(nifti_path, h5_path, dtype='int16'):
    # Load the .nii.gz file using nibabel
    nii_img = nib.load(nifti_path)
    nii_data = nii_img.get_fdata()
    
    #meta data
    nii_header = nii_img.header
    affine_matrix = nii_img.affine
    
    # Save the data and header to a .h5 file
    with h5py.File(h5_path, 'w') as h5f:
        h5f.create_dataset('nii_data', data=nii_data, compression='gzip', compression_opts=9, dtype=dtype)

        # write affine matrix to the root group
        h5f.attrs['affine_matrix'] = affine_matrix.tolist()

        # write header to the header group
        header_group = h5f.create_group('header')
        for key, value in nii_header.items():
            header_group.attrs[key] = value
            
def h5_to_nifti(h5_path, nifti_path):
    with h5py.File(h5_path, 'r') as h5f:
        data = np.array(h5f['nii_data'])
        header_group = h5f.get('header')
        header_dict = {key: header_group.attrs[key] for key in header_group.attrs.keys()}
        affine_matrix = np.array(h5f.attrs['affine_matrix'])

    header = nib.Nifti1Header()
    for key, value in header_dict.items():
        header[key] = value
        
    # Create nii image
    nii_img = nib.Nifti1Image(data, affine=affine_matrix, header=header)

    # Save nii image
    nib.save(nii_img, nifti_path)

    
def main(debug_mode=False):
    
    ORIGINAL_ROOT_DIR = '/Users/zongwei.zhou/Desktop/AbdomenAtlas_Core/'
    REVISED_ROOT_DIR = '/Users/zongwei.zhou/Desktop/AbdomenAtlas_Core_H5/'
    if not os.path.exists(REVISED_ROOT_DIR):
        os.makedirs(REVISED_ROOT_DIR)
    
    if debug_mode:
        REREVISED_ROOT_DIR = '/Users/zongwei.zhou/Desktop/AbdomenAtlas_Core_Nifti/'
        if not os.path.exists(REREVISED_ROOT_DIR):
            os.makedirs(REREVISED_ROOT_DIR)

    CT_ID_LIST = glob.glob(os.path.join(ORIGINAL_ROOT_DIR, '*'))
    CT_ID_LIST = [s.split('/')[-1] for s in CT_ID_LIST]

    for CT_ID in tqdm(CT_ID_LIST):

        ORIGINAL_CT_FILE = os.path.join(ORIGINAL_ROOT_DIR, CT_ID, 'ct.nii.gz')
        if not os.path.exists(os.path.join(REVISED_ROOT_DIR, CT_ID)):
            os.makedirs(os.path.join(REVISED_ROOT_DIR, CT_ID))
        REVISED_CT_FILE = os.path.join(REVISED_ROOT_DIR, CT_ID, 'ct.h5')
        nifti_to_h5(nifti_path=ORIGINAL_CT_FILE, h5_path=REVISED_CT_FILE, dtype='int16')

        

        ORIGINAL_MASK_FILE = glob.glob(os.path.join(ORIGINAL_ROOT_DIR, CT_ID, 'segmentations/*.nii.gz'))
        if not os.path.exists(os.path.join(REVISED_ROOT_DIR, CT_ID, 'segmentations')):
            os.makedirs(os.path.join(REVISED_ROOT_DIR, CT_ID, 'segmentations'))
        REVISED_MASK_FILE = [s.replace(ORIGINAL_ROOT_DIR, REVISED_ROOT_DIR) for s in ORIGINAL_MASK_FILE]
        REVISED_MASK_FILE = [s.replace('.nii.gz', '.h5') for s in REVISED_MASK_FILE]
        for orginal_mask, revised_mask in zip(ORIGINAL_MASK_FILE, REVISED_MASK_FILE):
            nifti_to_h5(nifti_path=orginal_mask, h5_path=revised_mask, dtype='int8')

        if debug_mode:
            if not os.path.exists(os.path.join(REREVISED_ROOT_DIR, CT_ID)):
                os.makedirs(os.path.join(REREVISED_ROOT_DIR, CT_ID))
            h5_to_nifti(h5_path=REVISED_CT_FILE, nifti_path=ORIGINAL_CT_FILE.replace(ORIGINAL_ROOT_DIR, REREVISED_ROOT_DIR))

            if not os.path.exists(os.path.join(REREVISED_ROOT_DIR, CT_ID, 'segmentations')):
                os.makedirs(os.path.join(REREVISED_ROOT_DIR, CT_ID, 'segmentations'))
            REREVISED_MASK_FILE = [s.replace(ORIGINAL_ROOT_DIR, REREVISED_ROOT_DIR) for s in ORIGINAL_MASK_FILE]
            for revised_mask, rerevised_mask in zip(REVISED_MASK_FILE, REREVISED_MASK_FILE):
                h5_to_nifti(h5_path=revised_mask, nifti_path=rerevised_mask)
        
if __name__ == "__main__":
    main(debug_mode=True)