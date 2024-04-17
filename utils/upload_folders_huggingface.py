'''
python upload_folders_huggingface.py
'''

from huggingface_hub import notebook_login
notebook_login()

import torch
import huggingface_hub
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd

# Base folder path on your computer
base_folder = Path("/data2/wenxuan/Project/BagofTricks/SuPreM/direct_inference/AbdomenAtlasDemo") # /path/to/the/folder
repository = "wenxuanchelsea/testAbdomenAtlas"  # Hugging face repository
repository_type = "dataset"  # can be dataset, model, or space
n = 1000  # number of files per commit

def get_all_files(root: Path):
    dirs = [root]
    while dirs:
        current_dir = dirs.pop()
        for candidate in current_dir.iterdir():
            if candidate.is_file():
                yield candidate
            elif candidate.is_dir():
                dirs.append(candidate)

def get_groups_of_n(n: int, iterator):
    assert n > 1
    buffer = []
    for elt in iterator:
        if len(buffer) == n:
            yield buffer
            buffer = []
        buffer.append(elt)
    if buffer:
        yield buffer

api = HfApi()

try:
    counter = torch.load('counter.pt')
except FileNotFoundError:
    counter = 0

for i, file_paths in enumerate(get_groups_of_n(n, get_all_files(base_folder))):
    if i < counter:
        continue
    print(f"Committing {i}")
    operations = [
        CommitOperationAdd(path_in_repo=str(file_path.relative_to(base_folder)),
                           path_or_fileobj=str(file_path))
        for file_path in file_paths
    ]
    api.create_commit(
        repo_id=repository,
        operations=operations,
        commit_message=f"Upload part {i}",
        repo_type=repository_type
    )
    torch.save(i, 'counter.pt')
