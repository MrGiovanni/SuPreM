import cc3d # pip install connected-components-3d
import numpy as np
from monai.data import DataLoader, Dataset, list_data_collate, DistributedSampler
from monai.config import DtypeLike, KeysCollection
from monai.transforms.transform import Transform, MapTransform
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

def rl_split(input_data, organ_index, right_index, left_index, name):
    '''
    input_data: 3-d tensor [w,h,d], after transform 'Orientationd(keys=["label"], axcodes="RAS")'
    oragn_index: the organ index of interest
    right_index and left_index: the corresponding index in template
    return [1, w, h, d]
    '''
    RIGHT_ORGAN = right_index
    LEFT_ORGAN = left_index
    label_raw = input_data.copy()
    label_in = np.zeros(label_raw.shape)
    label_in[label_raw == organ_index] = 1
    
    label_out = cc3d.connected_components(label_in, connectivity=26)
    # print('label_out', organ_index, np.unique(label_out), np.unique(label_in), label_out.shape, np.sum(label_raw == organ_index))
    # assert len(np.unique(label_out)) == 3, f'more than 2 component in this ct for {name} with {np.unique(label_out)} component'
    if len(np.unique(label_out)) > 3:
        count_sum = 0
        values, counts = np.unique(label_out, return_counts=True)
        num_list_sorted = sorted(values, key=lambda x: counts[x])[::-1]
        for i in num_list_sorted[3:]:
            label_out[label_out==i] = 0
            count_sum += counts[i]
        label_new = np.zeros(label_out.shape)
        for tgt, src in enumerate(num_list_sorted[:3]):
            label_new[label_out==src] = tgt
        label_out = label_new
        print(f'In {name}. Delete {len(num_list_sorted[3:])} small regions with {count_sum} voxels')
    a1,b1,c1 = np.where(label_out==1)
    a2,b2,c2 = np.where(label_out==2)
    
    label_new = np.zeros(label_out.shape)
    if np.mean(a1) < np.mean(a2):
        label_new[label_out==1] = LEFT_ORGAN
        label_new[label_out==2] = RIGHT_ORGAN
    else:
        label_new[label_out==1] = RIGHT_ORGAN
        label_new[label_out==2] = LEFT_ORGAN
    
    return label_new[None]


class RL_Split(Transform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, lbl: NdarrayOrTensor, name) -> NdarrayOrTensor:
        lbl_new = lbl.copy()

        organ_index = 12
        right_index = 12
        left_index = 13
        lbl_post = rl_split(lbl_new[0], organ_index, right_index, left_index, name)
        lbl_new[lbl_post == left_index] = left_index
        
        return lbl_new

class RL_Splitd(MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.spliter = RL_Split()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        d['label'] = self.spliter(d['label'], d['name'])
        return d