import torch
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from enum import Enum, auto


class DatasetType(Enum):
    TRAIN = auto()
    TEST = auto()


class MVTEC_AD(Dataset):
    r"""
    https://www.mvtec.com/company/research/datasets/mvtec-ad
    """

    class DataClass(Enum):
        Bottle = auto()
        Cable = auto()
        Capsule = auto()
        Carpet = auto()
        Grid = auto()
        Hazelnut = auto()
        Leather = auto()
        Metal_Nut = auto()
        Pill = auto()
        Screw = auto()
        Tile = auto()
        Toothbrush = auto()
        Transistor = auto()
        Wood = auto()
        Zipper = auto()

    NORMAL_OBJECTS_NAME = 'good'

    def __init__(self, dataset_type: DatasetType, data_type: DataClass, data_path: Path,
                 transform=None, mask_transform=None):
        data_path = data_path / data_type.name.lower()
        images_path = data_path / dataset_type.name.lower()
        masks_path = (data_path / 'ground_truth')
        self.is_train = dataset_type
        self.class_names = [MVTEC_AD.NORMAL_OBJECTS_NAME] + [p.stem for p in images_path.glob('**/')
                                                             if (p.is_dir() and p.stem != MVTEC_AD.NORMAL_OBJECTS_NAME)]

        self.images = []
        self.labels = []
        self.masks = []

        for cls, name in enumerate(self.class_names):
            images_fnames = [p for p in (images_path / name).glob('*.png')]
            for img_fn in images_fnames:
                self.labels.append(cls)
                # Please, forgive for this :c
                img = Image.open(str(img_fn))

                if self.is_train == DatasetType.TEST:
                    if name == MVTEC_AD.NORMAL_OBJECTS_NAME:
                        mask = Image.new('L', img.size, 0)
                    else:
                        mask = Image.open(str(masks_path / name / f'{img_fn.stem}_mask.png'))

                    if mask_transform is not None:
                        mask = mask_transform(mask)

                    self.masks.append(mask)

                if transform is not None:
                    img = transform(img)

                self.images.append(img)

    def __len__(self):
        return len(self.labels)

    @property
    def num_of_classes(self):
        return len(self.class_names)

    def class2name(self, cls: int):
        return self.class_names[cls]

    def name2class(self, name: str):
        return self.class_names.index(name)

    def __getitem__(self, idx):
        if self.is_train == DatasetType.TRAIN:
            return self.images[idx], self.labels[idx]
        return self.images[idx], self.labels[idx], self.masks[idx]
