'''
source /data/zzhou82/environments/syn/bin/activate
python -W ignore annotation_quality_assessment.py --datapath /Volumes/T9/AbdomenAtlasPro --aorta
'''

import os
import argparse
from helper_functions import *

def main(args):

    folder_names = [name for name in os.listdir(args.datapath) if os.path.isdir(os.path.join(args.datapath, name))]
    folder_names = sorted(folder_names)

    for pid in folder_names:

        if args.kidney:
            kidney_postprocessing(pid, args.datapath)
            

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', dest='datapath', type=str, default='/Volumes/T9/AbdomenAtlasPro',
                        help='the directory of the AbdomenAtlas dataset',
                       )
    parser.add_argument('--aorta', action='store_true', default=False, 
                        help='check label quality for aorta?',
                       )
    parser.add_argument('--kidney', action='store_true', default=False, 
                        help='check label quality for kidney?',
                       )
    args = parser.parse_args()
    
    main(args)