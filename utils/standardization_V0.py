'''bash
python standardization_V0.py --ROOT_DIR /Users/zongwei.zhou/Desktop/BodyMapsSmall --REVISED_DIR /Users/zongwei.zhou/Desktop/BodyMapsSmallUnique
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
    name = name.replace('18_FLARE23_FLARE23Ts_', '18_FLARE23_Ts_')
    
    return name

def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--ROOT_DIR', default=None, type=str, help='original root directory')
    parser.add_argument('--REVISED_DIR', default=None, type=str, help='revised root directory')
    
    args = parser.parse_args()

    assert args.ROOT_DIR is not None
    assert args.REVISED_DIR is not None

    os.makedirs(os.path.dirname(args.REVISED_DIR), exist_ok=True)

    CT_ID_LIST = glob.glob(os.path.join(args.ROOT_DIR, '*'))
    CT_ID_LIST = [os.path.basename(s) for s in CT_ID_LIST]

    for ct_id in tqdm(CT_ID_LIST):

        shutil.copytree(os.path.join(args.ROOT_DIR, ct_id), os.path.join(args.REVISED_DIR, rename_id(ct_id)))
        
if __name__ == "__main__":
    main()