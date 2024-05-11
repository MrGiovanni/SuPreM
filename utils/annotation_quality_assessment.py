'''
source /data/zzhou82/environments/syn/bin/activate
python -W ignore annotation_quality_assessment.py --datapath /Volumes/T9/AbdomenAtlasPro -o /Users/zongweizhou/Desktop/error_analysis --aorta --csvname aorta.csv
python -W ignore annotation_quality_assessment.py --datapath /Volumes/T9/AbdomenAtlasPro -o /Users/zongweizhou/Desktop/error_analysis --kidney --csvname kidney.csv
'''

import os
import argparse
import csv
from helper_functions import *

def error_detection_per_case(pid, args):

    if args.aorta:
        return aorta_error(pid, args.datapath)
    if args.kidney:
        return kidney_error(pid, args.datapath)

def main(args):

    if os.path.isfile(os.path.join(args.csvpath, args.csvname)):
        os.remove(os.path.join(args.csvpath, args.csvname))
    if not os.path.exists(args.csvpath):
        os.makedirs(args.csvpath)
    with open(os.path.join(args.csvpath, args.csvname), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Patient ID'])

    folder_names = [name for name in os.listdir(args.datapath) if os.path.isdir(os.path.join(args.datapath, name))]
    folder_names = sorted(folder_names)
    error_list = []

    for pid in folder_names:

        error_detected = error_detection_per_case(pid, args)
        if error_detected:
            error_list.append(pid)
            
            with open(os.path.join(args.csvpath, args.csvname), 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([pid])

    print('\n> Overall error report {:.1f}% = {}/{}'.format(100.0*len(error_list)/len(folder_names), len(error_list), len(folder_names)))
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', dest='datapath', type=str, default='/Volumes/T9/AbdomenAtlasPro',
                        help='the directory of the AbdomenAtlas dataset',
                       )
    parser.add_argument('-o', dest='csvpath', type=str, default='/Users/zongwei.zhou/Desktop/error_analysis',
                        help='the directory of the saved csv file recording all the error cases',
                       )
    parser.add_argument('--csvname', dest='csvname', type=str, default='aorta.csv',
                        help='the directory of the saved csv file recording all the error cases',
                       )
    parser.add_argument('--aorta', action='store_true', default=False, 
                        help='check label quality for aorta?',
                       )
    parser.add_argument('--kidney', action='store_true', default=False, 
                        help='check label quality for kidney?',
                       )
    args = parser.parse_args()
    
    main(args)