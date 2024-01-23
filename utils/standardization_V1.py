'''bash
python standardization_V1.py --ROOT_DIR /Users/zongwei.zhou/Desktop/BodyMapsSmall --ROOT_UNIQUE_DIR /Users/zongwei.zhou/Desktop/BodyMapsSmallUnique
'''

import os, shutil
import nibabel as nib
import numpy as np
import shutil
import glob
import argparse
from tqdm import tqdm
from skimage.measure import label

def rename_id(name):
    name = name.replace('01_Multi-Atlas_Labeling', '01_BTCV')
    name = name.replace('18_FLARE23_FLARE23_', '18_FLARE23_')
    name = name.replace('KiTS23_case_', '05_KiTS_case_')
    
    return name

# def compare_files_size(file1, file2): 

# 	if abs(os.path.getsize(file1) - os.path.getsize(file2)) < 100:    
# 		return True   
# 	else:
# 		return False

def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--ROOT_DIR', default=None, type=str, help='original root directory')
    parser.add_argument('--ROOT_UNIQUE_DIR', default=None, type=str, help='revised root directory')
    
    args = parser.parse_args()

    assert args.ROOT_DIR is not None
    assert args.ROOT_UNIQUE_DIR is not None

    os.makedirs(os.path.dirname(args.ROOT_UNIQUE_DIR), exist_ok=True)

    CT_ID_LIST = glob.glob(os.path.join(args.ROOT_DIR, '*'))
    CT_ID_LIST = [os.path.basename(s) for s in CT_ID_LIST]
    dic = []

    for ct_id in tqdm(CT_ID_LIST):
        if os.path.exists(os.path.join(args.ROOT_DIR, ct_id, 'segmentations')):
            img = nib.load(os.path.join(args.ROOT_DIR, ct_id, 'segmentations/liver.nii.gz'))
        else:
            img = nib.load(os.path.join(args.ROOT_DIR, ct_id, 'ct.nii.gz'))
        data = np.array(img.dataobj)

        duplication = False
        for i in range(len(dic)):
            if data.shape == dic[i]['dimension']:

                ct1 = nib.load(os.path.join(args.ROOT_DIR, ct_id, 'ct.nii.gz'))
                ct1 = np.array(ct1.dataobj)
                ct2 = nib.load(os.path.join(args.ROOT_UNIQUE_DIR, rename_id(dic[i]['ID']), 'ct.nii.gz'))
                ct2 = np.array(ct2.dataobj)
                if abs(np.sum(ct1 > 0) - np.sum(ct2 > 0)) < 20:
                    print('{} = {} = {}'.format(ct_id, dic[i]['ID'], 
                                                abs(np.sum(ct1 > 0) - np.sum(ct2 > 0))
                                               ))
                    duplication = True
                    break

        if duplication:
            continue
        else:
            case = {'ID': ct_id, 'dimension': data.shape}
            dic.append(case)
            shutil.move(os.path.join(args.ROOT_DIR, ct_id), os.path.join(args.ROOT_UNIQUE_DIR, rename_id(ct_id)))
        
if __name__ == "__main__":
    main()