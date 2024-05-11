'''
source /data/zzhou82/environments/syn/bin/activate
python -W ignore annotation_transfer.py --source_datapath /Volumes/T9/HGFC_inference_separate_data --destination_datapath /Volumes/T9/AbdomenAtlasPro -o error_analysis/aorta.csv -c aorta
python -W ignore annotation_transfer.py --source_datapath /Volumes/T9/HGFC_inference_separate_data --destination_datapath /Volumes/T9/AbdomenAtlasPro -o error_analysis/kidney.csv -c kidney_left
python -W ignore annotation_transfer.py --source_datapath /Volumes/Expansion/AbdomenAtlas/AbdomenAtlasPro --destination_datapath /Volumes/T9/AbdomenAtlasPro -o /Volumes/T9/error_analysis/aorta.csv -c aorta
'''

import os
import argparse
import csv
import shutil
from helper_functions import *

def main(args):

    error_id_list = []

    with open(args.csvpath, newline='') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            error_id_list.append(row[0])

    for pid in error_id_list:
        source_path = os.path.join(args.source_datapath, pid, 'segmentations', args.class_name + '.nii.gz')
        destination_path = os.path.join(args.destination_datapath, pid, 'segmentations', args.class_name + '.nii.gz')
        if os.path.isfile(source_path):
            shutil.copy(source_path, destination_path)
        else:
            print('>> Error: no such file {}'.format(source_path))
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_datapath', dest='source_datapath', type=str, default='/Volumes/T9/HGFC_inference_separate_data',
                        help='the directory of the AbdomenAtlas dataset',
                       )
    parser.add_argument('--destination_datapath', dest='destination_datapath', type=str, default='/Volumes/T9/AbdomenAtlasPro',
                        help='the directory of the AbdomenAtlas dataset',
                       )
    parser.add_argument('-o', dest='csvpath', type=str, default='error_analysis/aorta.csv',
                        help='the directory of the saved csv file recording all the error cases',
                       )
    parser.add_argument('-c', dest='class_name', type=str, default='aorta',
                        help='the class name in error cases',
                       )
    args = parser.parse_args()
    
    main(args)