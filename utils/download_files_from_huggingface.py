'''
python download_files_from_huggingface.py --repo_id MrGiovanni/BodyMaps --output_dir /Users/zongwei.zhou/Desktop/BodyMaps
'''

import os
from tqdm import tqdm
import argparse
from huggingface_hub import HfApi, hf_hub_download
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

def download_file(dataset_file, output_dir, repo_id, repo_type):
    '''
    This function downloads a file from the Hugging Face Hub using the hf_hub_download function.
    
    Args:
    - dataset_file: The file to download
    - output_dir: The directory to save the file to
    - repo_id: The repository ID to download from
    - repo_type: The type of repository to download from (e.g. dataset, space)
    '''
    try:
        # Download using hf_hub_download
        downloaded_file = hf_hub_download(repo_id=repo_id, filename=dataset_file.rfilename, repo_type=repo_type)

        # Construct the output path 
        folder = os.path.dirname(dataset_file.rfilename)  # Extract folder
        filename = os.path.basename(dataset_file.rfilename)  # Extract filename
        output_path = os.path.join(output_dir, folder, filename)

        # Ensure parent directories exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Read the downloaded file's content
        with open(downloaded_file, 'rb') as f:  
            file_bytes = f.read() 

        # Write the file to disk
        with open(output_path, 'wb') as f: 
            f.write(file_bytes)

    except Exception as e:
        print(f"Error downloading {dataset_file.rfilename}: {e}") # Use rfilename for clarity

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='Download files from Hugging Face dataset repo.')
    parser.add_argument('--repo_id', type=str, required=True,
                        help='The ID of the Hugging Face dataset repository')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='The directory where downloaded files will be saved')
    parser.add_argument('--repo_type', type=str, default='dataset',
                        help='The type of repository to download from (e.g. dataset, space)')

    args = parser.parse_args()

    # Get dataset info
    api = HfApi()
    dataset_info = api.dataset_info(repo_id=args.repo_id)
    files = dataset_info.siblings
    
    print('>> {} CPU cores are secured.'.format(cpu_count()))

    # Download files in parallel
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        futures = {executor.submit(download_file, file, args.output_dir, args.repo_id, args.repo_type): file for file in tqdm(files)}
        for future in tqdm(as_completed(futures), total=len(futures)):
            file_id = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {file_id}: {e}")
