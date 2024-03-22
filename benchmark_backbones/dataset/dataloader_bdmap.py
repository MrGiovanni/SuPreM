from monai.transforms import (
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    SpatialPadd,
)

import sys
import nibabel as nib
import os

import numpy as np
from typing import Optional, Union

sys.path.append("..") 

from monai.data import DataLoader, Dataset, list_data_collate, DistributedSampler, CacheDataset, SmartCacheDataset
from monai.config import DtypeLike, KeysCollection
from monai.transforms.transform import MapTransform
from monai.transforms.io.array import LoadImage
from monai.utils import ensure_tuple, ensure_tuple_rep
from monai.data.image_reader import ImageReader
from monai.utils.enums import PostFix
DEFAULT_POST_FIX = PostFix.meta()

class_map_abdomenatlas_1_0 = {
    0: "aorta",
    1: "gall_bladder",
    2: "kidney_left",
    3: "kidney_right",
    4: "liver",
    5: "pancreas",
    6: "postcava",
    7: "spleen",
    8: "stomach"
}

class LoadSelectedImaged(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        reader: Optional[Union[ImageReader, str]] = None,
        dtype: DtypeLike = np.int16,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        overwriting: bool = False,
        image_only: bool = False,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self._loader = LoadImage(reader, image_only, dtype, ensure_channel_first, simple_keys, *args, **kwargs)
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting


    def register(self, reader: ImageReader):
        self._loader.register(reader)


    def __call__(self, data, reader: Optional[ImageReader] = None):
        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            data = self._loader(d[key], reader)
            if self._loader.image_only:
                d[key] = data
            else:
                if not isinstance(data, (tuple, list)):
                    raise ValueError("loader must return a tuple or list (because image_only=False was used).")
                d[key] = data[0]
                if not isinstance(data[1], dict):
                    raise ValueError("metadata must be a dict.")
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key in d and not self.overwriting:
                    raise KeyError(f"Metadata with key {meta_key} already exists and overwriting=False.")
                d[meta_key] = data[1]
        label_parent_path = d['label_parent']
        label_organs = class_map_abdomenatlas_1_0
        temp = nib.load(os.path.join(label_parent_path,label_organs[0]+'.nii.gz')).get_fdata()
        W,H,D = temp.shape
        label = np.zeros((len(label_organs),W,H,D))
        
        for organ in range(len(label_organs)):
            selected_organ = label_organs[organ]
            organ_data = nib.load(os.path.join(label_parent_path, selected_organ+'.nii.gz')).get_fdata()
            label[organ][organ_data == 1] = 1
    
        d['label'] = label
        return d

def get_loader(args):
    train_transforms = Compose(
        [
            LoadSelectedImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z), mode='constant'),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=5,
                neg=1,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
            ), 
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.20,
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )
    ## training dict part
    train_img = []
    train_lbl_parents = []
    train_name = []

    for item in args.dataset_list:
        for line in open(os.path.join(args.data_txt_path,item +'.txt')):
            name = line.strip().split('\t')[0]
            train_img_path = os.path.join(args.data_root_path, name, 'ct.nii.gz')
            folder_name = os.path.join(args.data_root_path, name, 'segmentations/')
            train_img.append(train_img_path)
            train_lbl_parents.append(folder_name)
            train_name.append(name)

    data_dicts_train = [{'image': image, 'label_parent': label, 'name': name}
                for image, label,name in zip(train_img, train_lbl_parents,train_name)]
    print('train len {}'.format(len(data_dicts_train)))

    if args.cache_dataset:
        # train_dataset = CacheDataset(data=data_dicts_train, 
        #                              transform=train_transforms, 
        #                              cache_rate=args.cache_rate,
        #                             )
        train_dataset = SmartCacheDataset(
                                          data=data_dicts_train,
                                          transform=train_transforms,
                                          cache_num=args.cache_num,
                                          cache_rate=args.cache_rate,
                                         )
    else:   
        train_dataset = Dataset(data=data_dicts_train, transform=train_transforms)
    train_sampler = DistributedSampler(dataset=train_dataset, even_divisible=True, shuffle=True) if args.dist else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers, 
                                collate_fn=list_data_collate, sampler=train_sampler)
    return train_loader, train_sampler
