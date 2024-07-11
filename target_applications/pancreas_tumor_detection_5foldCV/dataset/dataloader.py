from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    ToTensord,
    SpatialPadd,
)

import sys
import os


import numpy as np
from typing import Optional, Union

sys.path.append("..") 

from monai.data import DataLoader, Dataset, list_data_collate, DistributedSampler
from monai.config import DtypeLike, KeysCollection
from monai.transforms.transform import MapTransform
from monai.transforms.io.array import LoadImage
from monai.utils import ensure_tuple, ensure_tuple_rep
from monai.data.image_reader import ImageReader
from monai.utils.enums import PostFix
DEFAULT_POST_FIX = PostFix.meta()

class_map_jhh={
    1: 'pancreas',
    2: 'pdac',
    3: 'cyst',
    4: 'pnet',
}

taskmap_set = {
    'jhh': class_map_jhh,
}

class LoadImaged_totoalseg(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        map_type,
        reader: Optional[Union[ImageReader, str]] = None,
        dtype: DtypeLike = np.float32,
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
        self.map_type = map_type


    def register(self, reader: ImageReader):
        self._loader.register(reader)


    def __call__(self, data, reader: Optional[ImageReader] = None):
        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            try:
                data = self._loader(d[key], reader)
            except:
                print(d['name'])
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
        d['label'], d['label_meta_dict'] = self.label_transfer(d['label'], self.map_type, d['image'].shape)
        
        return d

    def label_transfer(self, lbl_dir, map_type, shape):
        anatomical_structures_map = taskmap_set[map_type]
        anatomical_structure_lbl = np.zeros(shape)
        for index, anatomical_structure in anatomical_structures_map.items():
            array, mata_infomation = self._loader(lbl_dir + anatomical_structure + '.nii.gz')
            anatomical_structure_lbl[array == 1] = index
        
        return anatomical_structure_lbl, mata_infomation

def get_loader(args):
    train_transforms = Compose(
        [
            LoadImaged_totoalseg(keys=["image"], map_type=args.map_type),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.8,
                max_k=3,
            ),
            RandFlipd(
                keys=["image", "label"],
                prob=0.4,
                spatial_axis=None,
            ),
            SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z), mode='constant'),
            RandCropByPosNegLabeld(# modify original code...
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=3,
                neg=1,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
            ), # 8
            RandShiftIntensityd(
                keys=["image"],
                prob=0.5,
                offsets=0.10,
            ),
            RandGaussianSmoothd(
                keys=["image"],
                prob=0.2,
                sigma_x=(0.5, 1.15),
                sigma_y=(0.5, 1.15),
                sigma_z=(0.5, 1.15),
            ),
            RandGaussianNoised(
                keys=["image"],
                prob=0.2,
                mean=0.0, 
                std=0.01,
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )
    
    test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                AddChanneld(keys=["image"]),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=args.a_min,
                    a_max=args.a_max,
                    b_min=args.b_min,
                    b_max=args.b_max,
                    clip=True,
                ),
                ToTensord(keys=["image"]),
            ]
    )
    
    if args.stage == 'train':
        train_img = []
        train_lbl = []
        train_name = []
        for item in args.dataset_list:
            for line in open(os.path.join(args.data_txt_path,item +'.txt')):
                name = line.strip().split('\t')[0]
                train_img_path = os.path.join(args.dataset_path, name, 'ct.nii.gz')
                if not os.path.isfile(train_img_path):
                    print('{}: ct not exist!'.format(train_img_path))
                    raise
                train_lbl_path = os.path.join(args.dataset_path, name, 'segmentations/')
                if not os.path.isdir(train_lbl_path):
                    print('{}: segmentations not exist!'.format(train_lbl_path))
                    raise
                train_img.append(train_img_path)
                train_lbl.append(train_lbl_path)
                train_name.append(name)
        data_dicts_train = [{'image': image, 'label': label, 'name': name}
                    for image, label, name in zip(train_img, train_lbl, train_name)]
        # print('train len {}'.format(len(data_dicts_train)))

        train_dataset = Dataset(data=data_dicts_train, transform=train_transforms)
        train_sampler = DistributedSampler(dataset=train_dataset, even_divisible=True, shuffle=True) if args.dist else None
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers, 
                                    collate_fn=list_data_collate, sampler=train_sampler)
        return train_loader, train_sampler  
    
    if args.stage == 'test':
        test_img = []
        test_name_img=[]
        for item in args.dataset_list:
            item = item.split('.')[0]
            test_txt_path = os.path.join(args.data_txt_path, item + '.txt')
            for line in open(test_txt_path):
                name_img = line.strip().split('\t')[0]
                test_img.append(os.path.join(args.data_root_path, name_img, 'ct.nii.gz'))
                test_name_img.append(name_img)
        data_dicts_test = [{'image': image,'name_img':name_img}
                    for image, name_img in zip(test_img, test_name_img)]
        # print('test len {}'.format(len(data_dicts_test)))
        
        test_dataset = Dataset(data=data_dicts_test, transform=test_transforms)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=list_data_collate)
        return test_loader, test_transforms
