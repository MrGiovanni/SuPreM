'''bash
python get_dataset_statistics.py --datapath /Users/zongwei.zhou/Dropbox\ \(ASU\)/PublicResource/SuPreM/AbdomenAtlas/AbdomenAtlasDemo
'''

import os
import argparse

import nibabel as nib

from tqdm import tqdm

def main():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--datapath', default=None, type=str, help='data directory')

    args = parser.parse_args()

    assert args.datapath is not None

    folder_names = [name for name in os.listdir(args.datapath) if os.path.isdir(os.path.join(args.datapath, name))]
    num_slices = 0

    for pid in tqdm(folder_names):
        mask_path = os.path.join(args.datapath, pid, 'segmentations', 'liver.nii.gz')
        mask = nib.load(mask_path)
        mask_shape = mask.header['dim']
        if mask_shape[3] == 512 and mask_shape[1] != 512:
            num_slices += mask_shape[1]
        else:
            num_slices += mask_shape[3]
    print('>> Total number of slices = {}'.format(num_slices))

if __name__ == "__main__":
    main()