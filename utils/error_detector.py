'''
source /data/zzhou82/environments/syn/bin/activate
python -W ignore error_detector.py --datapath /Users/zongwei.zhou/Dropbox (ASU)/PublicResource/SuPreM/AbdomenAtlas/AbdomenAtlasProDemo
'''

import os
import argparse

import nibabel as nib
import numpy as np
import csv

from tqdm import tqdm


def main(args):

    folder_names = [name for name in os.listdir(args.datapath) if os.path.isdir(os.path.join(args.datapath, name))]
    folder_names = sorted(folder_names)
    error_list = []

    for pid in folder_names:

        mask_path = os.path.join(args.datapath, pid, 'segmentations', 'liver.nii.gz')
        liver_mask = nib.load(mask_path).get_fdata().astype(np.uint8)

        mask_path = os.path.join(args.datapath, pid, 'segmentations', 'aorta.nii.gz')
        aorta_mask = nib.load(mask_path).get_fdata().astype(np.uint8)

        mask_path = os.path.join(args.datapath, pid, 'segmentations', 'lung_left.nii.gz')
        lung_left_mask = nib.load(mask_path).get_fdata().astype(np.uint8)

        mask_path = os.path.join(args.datapath, pid, 'segmentations', 'lung_right.nii.gz')
        lung_right_mask = nib.load(mask_path).get_fdata().astype(np.uint8)

        mask_path = os.path.join(args.datapath, pid, 'segmentations', 'postcava.nii.gz')
        postcava_mask = nib.load(mask_path).get_fdata().astype(np.uint8)

        error_count = 0

        if (liver_mask.shape[0] == 512 and liver_mask.shape[1] == 512) or \
        (liver_mask.shape[0] != liver_mask.shape[1] and liver_mask.shape[1] != liver_mask.shape[2]):
            for i in range(liver_mask.shape[-1]):
                if (np.sum(liver_mask[:,:,i]) > 0 and np.sum(aorta_mask[:,:,i]) == 0) or \
                (np.sum(postcava_mask[:,:,i]) > 0 and (np.sum(lung_right_mask[:,:,i]) > 0 or np.sum(lung_left_mask[:,:,i]) > 0) and np.sum(aorta_mask[:,:,i]) == 0):
                    error_count += 1
            print('> {} has {:.1f}% ({}/{}) errors'.format(pid,
                                                           100.0*error_count/liver_mask.shape[-1],
                                                           error_count,
                                                           liver_mask.shape[-1],
                                                           ))
            
            if error_count/liver_mask.shape[-1] > 0.03:
                error_percent = round(100.0*error_count/liver_mask.shape[-1], 1)
                total_slices = liver_mask.shape[-1]
                error_list.append((pid, error_percent, error_count, total_slices))
        else:
            for i in range(liver_mask.shape[0]):
                if (np.sum(liver_mask[i,:,:]) > 0 and np.sum(aorta_mask[i,:,:]) == 0) or \
                (np.sum(postcava_mask[i,:,:]) > 0 and (np.sum(lung_right_mask[i,:,:]) > 0 or np.sum(lung_left_mask[i,:,:]) > 0) and np.sum(aorta_mask[i,:,:]) == 0):
                    error_count += 1

            print('> {} has {:.1f}% ({}/{}) errors'.format(pid,
                                                           100.0*error_count/liver_mask.shape[0],
                                                           error_count,
                                                           liver_mask.shape[0],
                                                           ))
            
            if error_count/liver_mask.shape[0] > 0.03:
                error_percent = round(100.0*error_count/liver_mask.shape[0], 1)
                total_slices = liver_mask.shape[0]
                error_list.append((pid, error_percent, error_count, total_slices))

    print('\n> Overall error report {:.1f}% = {}/{}'.format(100.0*len(error_list)/len(folder_names), len(error_list), len(folder_names)))
    return error_list
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', dest='datapath', type=str, default='/Users/zongwei.zhou/Dropbox (ASU)/PublicResource/SuPreM/AbdomenAtlas/AbdomenAtlasProDemo',
                        help='the directory of the AbdomenAtlas dataset',
                       )
    args = parser.parse_args()
    
    error_list = main(args)
    
    with open('error_list.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Patient ID', 'Error Percentage', 'Error Count', 'Total Slices'])
        for pid, error_percent, error_count, total_slices in error_list:
            writer.writerow([pid, error_percent, error_count, total_slices])
    
    print('Error list saved to error_list.csv')
