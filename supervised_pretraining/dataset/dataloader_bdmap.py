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

# class map for the AbdomenAtlas 1.0 dataset
class_map_abdomenatlas_1_0 = {
    0: "aorta",
    1: "gall_bladder",
    2: "kidney_left",
    3: "kidney_right",
    4: "liver",
    5: "pancreas",
    6: "postcava",
    7: "spleen",
    8: "stomach",
    }

# class map for the AbdomenAtlas 1.1 dataset
class_map_abdomenatlas_1_1 = {
    0: 'aorta', 
    1: 'gall_bladder', 
    2: 'kidney_left', 
    3: 'kidney_right', 
    4: 'liver', 
    5: 'pancreas', 
    6: 'postcava', 
    7: 'spleen', 
    8: 'stomach', 
    9: 'adrenal_gland_left', 
    10: 'adrenal_gland_right', 
    11: 'bladder', 
    12: 'celiac_trunk', 
    13: 'colon', 
    14: 'duodenum', 
    15: 'esophagus', 
    16: 'femur_left', 
    17: 'femur_right', 
    18: 'hepatic_vessel', 
    19: 'intestine', 
    20: 'lung_left', 
    21: 'lung_right', 
    22: 'portal_vein_and_splenic_vein', 
    23: 'prostate', 
    24: 'rectum'
    }

abdomenatlas_set = {
    'AbdomenAtlas1.0': class_map_abdomenatlas_1_0,
    'AbdomenAtlas1.1': class_map_abdomenatlas_1_1,
}

class LoadSelectedImaged(MapTransform):
    """
    Custom transform to load a specific image and metadata using a flexible reader.

    Args:
        keys: Keys of the data dictionary to load selected images.
        reader: Image reader object or string reference.
        dtype: Data type for loaded images.
        meta_keys: Keys to store metadata along with image data.
        meta_key_postfix: Suffix for metadata keys.
        overwriting: Flag to allow overwriting existing metadata.
        image_only: Load only the image data (not metadata).
        ensure_channel_first: Reshape image into channel-first format if necessary.
        simple_keys: Use simplified, top-level data keys.
        allow_missing_keys: If True, missing data keys are ignored
    """
    def __init__(
        self,
        keys: KeysCollection,
        dataset_version,
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
        self.dataset_version = dataset_version


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
        
        # based on which dataset is being used, load the appropriate class map
        label_organs = abdomenatlas_set[self.dataset_version]
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
    """ 
    Creates training transforms, constructs a dataset, and returns a dataloader.

    Args:
        args: Command line arguments containing dataset paths and hyperparameters.
    """

    train_transforms = Compose(
        [
            LoadSelectedImaged(keys=["image"], dataset_version=args.dataset_version), # custom data loading 
            AddChanneld(keys=["image"]), # ensure a channel dimension
            Orientationd(keys=["image", "label"], axcodes="RAS"), # standardize orientation
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ), # adjust data spacing 
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ), # normalize intensity range
            CropForegroundd(keys=["image", "label"], source_key="image"), # crop to area of interest
            SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z), mode='constant'), # pad to standard size 
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=2,
                neg=1,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
            ), # random labeled crops
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ), # random rotations
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.20,
            ), # random intensity shifts
            ToTensord(keys=["image", "label"]), # convert to PyTorch tensors
        ]
    )
    
    # constructing training dataset
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

    # smart caching (if enabled)
    if args.cache_dataset:
        train_dataset = SmartCacheDataset(
                                          data=data_dicts_train,
                                          transform=train_transforms,
                                          cache_num=args.cache_num,
                                          cache_rate=args.cache_rate,
                                         )
    else:   
        train_dataset = Dataset(data=data_dicts_train, transform=train_transforms)
    
    # distributed sampler settings 
    train_sampler = DistributedSampler(dataset=train_dataset, even_divisible=True, shuffle=True) if args.dist else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers, 
                                collate_fn=list_data_collate, sampler=train_sampler)
    return train_loader, train_sampler
