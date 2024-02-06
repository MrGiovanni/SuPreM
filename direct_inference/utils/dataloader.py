from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    CenterSpatialCropd,
    Resized,
    SpatialPadd,
    apply_transform,
    RandZoomd,
    RandCropByLabelClassesd,
)

import sys
import os

import numpy as np
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

sys.path.append("..") 
import monai
from monai.data import DataLoader, Dataset, list_data_collate, DistributedSampler, CacheDataset
from monai.config import DtypeLike, KeysCollection
from monai.transforms.transform import Transform, MapTransform
from monai.transforms.io.array import LoadImage, SaveImage
from monai.utils import GridSamplePadMode, ensure_tuple, ensure_tuple_rep
from monai.data.image_reader import ImageReader
from monai.utils.enums import PostFix
from preprocess import RL_Splitd
DEFAULT_POST_FIX = PostFix.meta()

class_map_25organ_totalseg = {
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "esophagus",
    6: "liver",
    7: "stomach",
    8: "aorta",
    9: "inferior_vena_cava",
    10: "portal_vein_and_splenic_vein",
    11: "pancreas",
    12: "adrenal_gland_right",
    13: "adrenal_gland_left",
    14: "duodenum",
    16: "lung_right",
    17: "lung_left",
    18: "colon",
    21: "urinary_bladder",
    23: "femur_left",
    24: "femur_right",
}

class_map_25organ_jhh = {
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    6: "liver",
    7: "stomach",
    8: "aorta",
    9: "inferior_vena_cava",
    11: "pancreas",
    12: "adrenal_gland_right",
    13: "adrenal_gland_left",
    14: "duodenum",
    18: "colon",
    19: "intestine",
    25: "celiac_trunk",
}

class_map_25organ_atlas = {
    1: "spleen", # 26
    2: "kidney_right", # 16
    3: "kidney_left", # 15
    4: "gallbladder", # 12
    5: "esophagus", # 6
    6: "liver", # 13
    7: "stomach", # 7
    8: "aorta", # 114
    9: "inferior_vena_cava", # 117
    10: "portal_vein_and_splenic_vein", # 118
    11: "pancreas", # 14
    12: "adrenal_gland_right", # 28
    13: "adrenal_gland_left", # 27
    14: "duodenum", # 9
    16: "lung_right", # merge 122 123 124
    17: "lung_left", # merge 120 121
    18: "colon", # 10
    21: "urinary_bladder", # 17
    22: "prostate", # 19
    23: "femur_left", # 103
    24: "femur_right", # 104
    25: "celiac_trunk", # 119
}

taskmap_set = {
    'totalseg': class_map_25organ_totalseg,
    'jhh':class_map_25organ_jhh,
    'atlas':class_map_25organ_atlas
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
        organ_map = taskmap_set[map_type]
        organ_lbl = np.zeros(shape)
        # print('organ_map', organ_map)
        for index, organ in organ_map.items():
            if 'lung_left' == organ:
                array, mata_infomation = self._loader(lbl_dir + 'lung_upper_lobe_left' + '.nii.gz')
                organ_lbl[array == 1] = index
                array, mata_infomation = self._loader(lbl_dir + 'lung_lower_lobe_left' + '.nii.gz')
                organ_lbl[array == 1] = index

            elif 'lung_right' == organ:
                array, mata_infomation = self._loader(lbl_dir + 'lung_upper_lobe_right' + '.nii.gz')
                organ_lbl[array == 1] = index
                array, mata_infomation = self._loader(lbl_dir + 'lung_middle_lobe_right' + '.nii.gz')
                organ_lbl[array == 1] = index
                array, mata_infomation = self._loader(lbl_dir + 'lung_lower_lobe_right' + '.nii.gz')
                organ_lbl[array == 1] = index
                
            else:
                array, mata_infomation = self._loader(lbl_dir + organ + '.nii.gz')
                organ_lbl[array == 1] = index
        
        return organ_lbl, mata_infomation

class LoadImaged_jhh(MapTransform):
    def __init__(self,keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.reader1 = monai.transforms.LoadImaged(keys=keys, allow_missing_keys=allow_missing_keys)

    def __call__(self, data):
        d = dict(data)

        d = self.reader1.__call__(d)
        new_lbl = np.zeros(d['label'].shape) 
        new_lbl[d['label']==1] = 8
        new_lbl[d['label']==2] = 12
        new_lbl[d['label']==5] = 25
        new_lbl[d['label']==6] = 18
        new_lbl[d['label']==7] = 14
        new_lbl[d['label']==8] = 4
        new_lbl[d['label']==9] = 9
        new_lbl[d['label']==10] = 3
        new_lbl[d['label']==11] = 2
        new_lbl[d['label']==12] = 6
        new_lbl[d['label']==13] = 11
        new_lbl[d['label']==16] = 19
        new_lbl[d['label']==18] = 1
        new_lbl[d['label']==19] = 7
        d['label'] = new_lbl
        return d

class LoadImaged_atlas(MapTransform):
    def __init__(self,keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.reader1 = monai.transforms.LoadImaged(keys=keys, allow_missing_keys=allow_missing_keys)

    def __call__(self, data):
        d = dict(data)

        d = self.reader1.__call__(d)
        new_lbl = np.zeros(d['label'].shape) 
        new_lbl[d['label']==26] = 1
        new_lbl[d['label']==16] = 2
        new_lbl[d['label']==15] = 3
        new_lbl[d['label']==12] = 4
        new_lbl[d['label']==6] = 5
        new_lbl[d['label']==13] = 6
        new_lbl[d['label']==7] = 7
        new_lbl[d['label']==114] = 8
        new_lbl[d['label']==117] = 9
        new_lbl[d['label']==118] = 10
        new_lbl[d['label']==14] = 11
        new_lbl[d['label']==28] = 12
        new_lbl[d['label']==27] = 13
        new_lbl[d['label']==9] = 14
        new_lbl[d['label']==122] = 16
        new_lbl[d['label']==123] = 16
        new_lbl[d['label']==124] = 16
        new_lbl[d['label']==120] = 17
        new_lbl[d['label']==121] = 17
        new_lbl[d['label']==10] = 18
        new_lbl[d['label']==17] = 21
        new_lbl[d['label']==19] = 22
        new_lbl[d['label']==103] = 23
        new_lbl[d['label']==104] = 24
        new_lbl[d['label']==119] = 25
        d['label'] = new_lbl
        return d
    
class Load_totoalseg(MapTransform):
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
        d['label'], d['label_meta_dict'] = self.label_transfer(d['label'], self.map_type, d['pred'].shape)
        
        return d

    def label_transfer(self, lbl_dir, map_type, shape):
        organ_map = taskmap_set[map_type]
        organ_lbl = np.zeros(shape)
        # print('organ_map', organ_map)
        for index, organ in organ_map.items():
            if 'lung_left' == organ:
                array, mata_infomation = self._loader(lbl_dir + 'lung_upper_lobe_left' + '.nii.gz')
                organ_lbl[array == 1] = index
                array, mata_infomation = self._loader(lbl_dir + 'lung_lower_lobe_left' + '.nii.gz')
                organ_lbl[array == 1] = index

            elif 'lung_right' == organ:
                array, mata_infomation = self._loader(lbl_dir + 'lung_upper_lobe_right' + '.nii.gz')
                organ_lbl[array == 1] = index
                array, mata_infomation = self._loader(lbl_dir + 'lung_middle_lobe_right' + '.nii.gz')
                organ_lbl[array == 1] = index
                array, mata_infomation = self._loader(lbl_dir + 'lung_lower_lobe_right' + '.nii.gz')
                organ_lbl[array == 1] = index
                
            else:
                array, mata_infomation = self._loader(lbl_dir + organ + '.nii.gz')
                organ_lbl[array == 1] = index
        
        return organ_lbl, mata_infomation

def get_loader_totalseg(args):
    test_transforms = Compose(
        [
            LoadImaged_totoalseg(keys=["image"], map_type=args.map_type),
            AddChanneld(keys=["image", "label"]),
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
            CropForegroundd(keys=["image"], source_key="image"),
            ToTensord(keys=["image", "label"]),
        ]
    )
    
    
    ## test dict part
    test_img = []
    test_lbl = []
    test_name = []
    for line in open(args.data_txt_path + 'test_v201.txt'):
        name = line.strip().split('\t')[0]
        test_img.append(args.dataset_path + name + '/ct.nii.gz')
        test_lbl.append(args.dataset_path + name + '/segmentations/')
        test_name.append(name)
    data_dicts_test = [{'image': image, 'label': label, 'name': name}
                for image, label, name in zip(test_img, test_lbl, test_name)]
    print('test len {}'.format(len(data_dicts_test)))
    
    test_dataset = Dataset(data=data_dicts_test, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)

    return test_loader, test_transforms

def get_loader_jhh(args):
    test_transforms = Compose(
        [
            LoadImaged_jhh(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            RL_Splitd(keys=["label"]),
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
            CropForegroundd(keys=["image"], source_key="image"),
            ToTensord(keys=["image", "label"]),
        ]
    )
    
    
    ## test dict part
    test_img = []
    test_lbl = []
    test_name = []
    for line in open(args.data_txt_path + 'jhh_test.txt'):
        name = line.strip().split('\t')[0]
        if args.data_phase.upper() == 'VENOUS':
            test_img.append(args.dataset_path + 'img/' + name + '_VENOUS.nii.gz')
            test_lbl.append(args.dataset_path + 'label/' + name + '_VENOUS.nii.gz')
        elif args.data_phase.upper() == 'ARTERIAL':
            test_img.append(args.dataset_path + name + '_ARTERIAL.nii.gz')
            test_lbl.append(args.dataset_path + name + '_ARTERIAL.nii.gz')
        test_name.append(name)
    data_dicts_test = [{'image': image, 'label': label, 'name': name}
                for image, label, name in zip(test_img, test_lbl, test_name)]
    print('test len {}'.format(len(data_dicts_test)))
    data_dicts_test = data_dicts_test[:2]
    # breakpoint()
    test_dataset = Dataset(data=data_dicts_test, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)

    return test_loader, test_transforms

def path_loader_jhh(args):
    test_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(
                keys=["image"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear"),
            ), 
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            ToTensord(keys=["image"]),
        ]
    )
    
    
    ## test dict part
    test_img = os.listdir(args.dataset_path)[0]
    data_dicts_test = [{'image': os.path.join(args.dataset_path, test_img), "name": test_img.split('.')[0]}]
    print('data_dicts_test',data_dicts_test)
    test_dataset = Dataset(data=data_dicts_test, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)

    return test_loader, test_transforms

def get_loader_atlas(args):
    test_transforms = Compose(
        [
            LoadImaged_atlas(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
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
            ToTensord(keys=["image", "label"]),
        ]
    )
    
    ## test dict part
    test_img = []
    test_lbl = []
    test_name = []
    name_list = os.listdir(os.path.join(args.dataset_path, 'label'))
    name_list.sort()
    for line in name_list:
        name = line.split('.nii.gz')[0]
        test_img.append(args.dataset_path + 'CT/' + line)
        test_lbl.append(args.dataset_path + 'label/' +line)
        test_name.append(name)
    data_dicts_test = [{'image': image, 'label': label, 'name': name}
                for image, label, name in zip(test_img, test_lbl, test_name)]
    print('test len {}'.format(len(data_dicts_test)))
    # data_dicts_test = data_dicts_test[:1]

    test_dataset = Dataset(data=data_dicts_test, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)

    return test_loader, test_transforms

def eval_loader_jhh(args):
    test_transforms = Compose(
        [
            LoadImaged_jhh(keys=["pred", "label"]),
            # AddChanneld(keys=["pred", "label"]),
            Orientationd(keys=["pred", "label"], axcodes="RAS"),
            ToTensord(keys=["pred", "label"]),
        ]
    )
    
    ## test dict part
    test_pred = []
    test_lbl = []
    test_name = []
    for line in open(args.data_txt_path + 'jhh_test.txt'):
        name = line.strip().split('\t')[0]
        if args.data_phase.upper() == 'VENOUS':
            test_pred.append(args.pred_path + name + '_VENOUS_pred.nii.gz')
            test_lbl.append(args.dataset_path + 'label/' + name + '_VENOUS.nii.gz')
        elif args.data_phase.upper() == 'ARTERIAL':
            test_pred.append(args.pred_path + name + '_ARTERIAL_pred.nii.gz')
            test_lbl.append(args.dataset_path + 'label/' + name + '_ARTERIAL.nii.gz')
        test_name.append(name)
    data_dicts_test = [{'pred': pred, 'label': label, 'name': name}
                for pred, label, name in zip(test_pred, test_lbl, test_name)]
    print('test len {}'.format(len(data_dicts_test)))

    test_dataset = Dataset(data=data_dicts_test, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)

    return test_loader, test_transforms
    
def path_loader_totalseg(args):
    test_transforms = Compose(
        [
            Load_totoalseg(keys=["pred"], map_type=args.map_type),
            # AddChanneld(keys=["pred", "label"]),
            Orientationd(keys=["pred", "label"], axcodes="RAS"),
            ToTensord(keys=["pred", "label"]),
        ]
    )
    
    
    ## test dict part
    test_pred = []
    test_lbl = []
    test_name = []
    for line in open(args.data_txt_path + 'test.txt'):
        name = line.strip().split('\t')[0]
        test_pred.append(args.pred_path + name + '_pred.nii.gz')
        test_lbl.append(args.dataset_path + name + '/segmentations/')
        test_name.append(name)
    data_dicts_test = [{'pred': pred, 'label': label, 'name': name}
                for pred, label, name in zip(test_pred, test_lbl, test_name)]
    print('test len {}'.format(len(data_dicts_test)))

    test_dataset = Dataset(data=data_dicts_test, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)

    return test_loader, test_transforms