"""
Utilities

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Microsoft Research Asia
"""

import os
import time
import torch
import pickle
import numpy as np
import scipy.io as sio

try:
    import wandb
except ImportError:
    pass

from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset

from vcoco.vcoco import VCOCO
from hicodet.hicodet import HICODet

import pocket
from pocket.core import DistributedLearningEngine
from pocket.utils import DetectionAPMeter, BoxPairAssociation

from ops import recover_boxes
from detr.datasets import transforms as T


class DataFactoryOOD(Dataset):
    def __init__(self, name, partition, data_root):
        if name not in ['hicodet']:
            raise ValueError("Unknown dataset ", name)

        self.dataset = HICODet(
            root="/workspace/dataset/swig-hoi/images_512",
            anno_file="/workspace/dataset/swig-hoi/tmp/swig_hico.json",
            object_cls_num=80,
            verb_cls_num=305,
            hoi_cls_num=1161,
            target_transform=pocket.ops.ToTensor(input_format='dict')
        )

        # Prepare dataset transforms
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        if partition.startswith('train'):
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(),
                T.ColorJitter(.4, .4, .4),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales, max_size=1333),
                    ])
                ), normalize,
        ])
        else:
            self.transforms = T.Compose([
                T.RandomResize([800], max_size=1333),
                normalize,
            ])

        self.name = name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image, target = self.dataset[i]
        target['labels'] = target['verb']
        image, target = self.transforms(image, target)

        return image, target


if __name__ == "__main__":
    dataset = DataFactoryOOD(
        name="hicodet",
        partition="",
        data_root=""
    )
    print(len(dataset))


