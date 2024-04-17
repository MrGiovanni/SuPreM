'''
python upload_folders_huggingface.py
'''

from huggingface_hub import notebook_login
notebook_login()

import torch
import huggingface_hub
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd

folder = "AbdomenAtlasDemo" #path on your computer
repository = "MrGiovanni/AbdomenAtlasPro" #Hugging face repository
repository_type = "dataset" #dataset, model or space
n = 1000 #number of files per commit

def get_all_files(root: Path):
    dirs = [root]
    while len(dirs) > 0:
        dir = dirs.pop()
        for candidate in dir.iterdir():
            if candidate.is_file():
                yield candidate
            if candidate.is_dir():
                dirs.append(candidate)

def get_groups_of_n(n: int, iterator):
    assert n > 1
    buffer = []
    for elt in iterator:
        if len(buffer) == n:
            yield buffer
            buffer = []
        buffer.append(elt)
    if len(buffer) != 0:
        yield buffer


api = HfApi()
root = Path(folder)

try:
    counter=torch.load('counter.pt')
except:
    counter=0
#counter=0

for i, file_paths in enumerate(get_groups_of_n(n, get_all_files(root))):
    if i<counter:
        continue
    print(f"Committing {i}")
    operations = [
        CommitOperationAdd(path_in_repo='/'.join(str(file_path).split('/')[2:]), 
                           path_or_fileobj=str(file_path))
        for file_path in file_paths
    ]
    api.create_commit(
        repo_id=repository,
        operations=operations,
        commit_message=f"Upload part {i}",
        repo_type=repository_type
    )
    torch.save(i,'counter.pt')
