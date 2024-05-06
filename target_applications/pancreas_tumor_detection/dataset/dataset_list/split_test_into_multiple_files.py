'''
python split_test_into_multiple_files.py --file_path /data2/wenxuan/Project/J/dataset/dataset_list/jhh_test.txt --save_dir ./temp --lines_per_file 30 --file_suffix subset --extension .txt
'''

import argparse
import os

def split_file(args):
    # Open the original file and read the lines
    with open(args.file_path, 'r') as file:
        lines = file.readlines()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # Initialize variables to track file subsets and line count
    subset_number = 1
    line_count = 0
    
    # find the base of file path for file name
    file_name = os.path.basename(args.file_path).split(".")[0]

    # Prepare to write to the first file subset
    subset_file_path = f"{file_name}_{args.file_suffix}_{subset_number}{args.extension}"
    subset_file_path = os.path.join(args.save_dir, subset_file_path)
    subset_file = open(subset_file_path, "w")

    # Go through all lines from the original file and write them to subset files
    for line in lines:
        # If we hit the line limit, start a new file subset
        if line_count == args.lines_per_file:
            subset_file.close()
            subset_number += 1
            subset_file_path = f"{file_name}_{args.file_suffix}_{subset_number}{args.extension}"
            subset_file_path = os.path.join(args.save_dir, subset_file_path)
            subset_file = open(subset_file_path, "w")
            line_count = 0
        
        # Write the current line to the current subset file
        subset_file.write(line)
        line_count += 1

    # Close the last subset file
    subset_file.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', required=True, help='Path to the file to split')
    parser.add_argument('--save_dir', required=True, help='Directory to save the split files')
    parser.add_argument('--lines_per_file', type=int, default=30, help='Number of lines per file')
    parser.add_argument('--file_suffix', default='subset', help='Suffix for the split files')
    parser.add_argument('--extension', default='.txt', help='Extension for the split files')
    args = parser.parse_args()
    
    split_file(args)

if __name__ == "__main__":
    main()
    
    
