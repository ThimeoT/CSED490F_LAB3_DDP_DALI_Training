from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import torch
import torch.distributed as dist
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
import os
from math import ceil
try:
    import nvidia.dali as dali
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali import ops as ops
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

base_dir="/SSD/CIFAR"

classes = (
    'plane',
    'car',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck')

# DALI uses uint8 range [0, 255]
CIFAR_MEAN = [0.4913999 * 255, 0.48215866 * 255, 0.44653133 * 255]
CIFAR_STD = [0.24703476 * 255, 0.24348757 * 255, 0.26159027 * 255]

def fn_dali_cutout(images, cutout_length):
    side = float(cutout_length) / 32.0
    ax = fn.random.uniform(range=(0.0, 1.0 - side))
    ay = fn.random.uniform(range=(0.0, 1.0 - side))
    anchor = fn.stack(ay, ax)
    # CHW -> erase over H,W => axes=(1,2)
    return fn.erase(images,
                    anchor=anchor, shape=[side, side],
                    normalized_anchor=True, normalized_shape=True,
                    axes=(1, 2), fill_value=0.0)

class DALIWrapper:
    def __init__(self, dali_iter):
        self.dali_iter = dali_iter
    def __iter__(self):
        return self
    def __next__(self):
        data = self.dali_iter.__next__()[0]
        return data['data'], data['label'].squeeze(-1).long()
    def __len__(self):
        return ceil(self.dali_iter.size / self.dali_iter.batch_size)
    def reset(self):
        self.dali_iter.reset()

class CifarPipeline(Pipeline):
    '''Problem 6:
    Implement a DALI pipeline for CIFAR-10 dataset.
    As it different from the DDP loader, you need to implement data augmentation process in DALI pipeline style.
    Refer the following DDP transform.
    # DDP transform style
    transform_train = \
        transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4913999, 0.48215866, 0.44653133),
                (0.24703476, 0.24348757, 0.26159027))
            ])

    transform_test = \
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4913999, 0.48215866, 0.44653133),
                (0.24703476, 0.24348757, 0.26159027))
            ])
    '''

    def __init__(self, data_dir, batch_size, is_train, cutout_length,
                 device_id, shard_id, num_shards, num_workers):
        super(CifarPipeline, self).__init__(batch_size, num_workers, device_id, seed=12345)
        self.data_dir = data_dir
        self.is_train = is_train
        self.cutout_length = cutout_length
        self.shard_id = shard_id
        self.num_shards = num_shards

    def define_graph(self):
        images, labels = fn.readers.file(
            file_root=self.data_dir,
            random_shuffle=self.is_train,
            shard_id=self.shard_id,
            num_shards=self.num_shards,
            pad_last_batch=True,
            name="Reader",
        )
        images = fn.decoders.image(
            images, device="mixed", output_type=types.RGB
        )

        if self.is_train:
            # 1. Padding (pad 4px on each side -> 32x32 becomes 40x40,
            #    placing the original image at a random position in the
            #    padded canvas, equivalent to RandomCrop(32, padding=4))
            images = fn.paste(
                images,
                fill_value=0,
                ratio=1.25,
                paste_x=fn.random.uniform(range=(0.0, 1.0)),
                paste_y=fn.random.uniform(range=(0.0, 1.0)),
            )
            # 2. Horizontal Flip
            coin = fn.random.coin_flip(probability=0.5)
            # 3. Crop, Mirror, Normalize
            images = fn.crop_mirror_normalize(
                images,
                crop=(32, 32),
                mirror=coin,
                mean=CIFAR_MEAN,
                std=CIFAR_STD,
                output_layout="CHW",
                dtype=types.FLOAT,
            )
            # 4. Cutout
            if self.cutout_length > 0:
                images = fn_dali_cutout(images, self.cutout_length)
        else:
            # 1. Crop, Normalize
            images = fn.crop_mirror_normalize(
                images,
                crop=(32, 32),
                mean=CIFAR_MEAN,
                std=CIFAR_STD,
                output_layout="CHW",
                dtype=types.FLOAT,
            )

        labels = labels.gpu()
        return images, labels

def get_DALI_loader(test_batch, train_batch, root=base_dir, valid_size=0, valid_batch=0,
               cutout=16, num_workers=4, download=True, random_seed=12345, shuffle=True):
    ''' Problem 7: Get DALI loader
    (./handler/DALI/cifar10_loader.py)
    Implement get_DALI_loader function.
    get_DALI_loader function is used to get DALI loader.
    Because DALI loader is slightly different with DP/DDP loader, you may change few parts of get_DP_loader.

    DALIGenericIterator loads data slightly different from the original loader.
    ex) data['data'], data['label'].squeeze(-1).long()
    We give you the DALIWrapper class, which is a wrapper of DALIGenericIterator, to maintain the same interface with the original loader.
    '''

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device_id = torch.cuda.current_device()

    train_dir = os.path.join(root, "train")
    valid_dir = os.path.join(root, "valid")
    test_dir = os.path.join(root, "test")

    train_loader, valid_loader, test_loader = None, None, None

    if train_batch > 0:
        pipe = CifarPipeline(
            data_dir=train_dir, batch_size=train_batch, is_train=True,
            cutout_length=cutout, device_id=device_id, shard_id=rank,
            num_shards=world_size, num_workers=num_workers)
        pipe.build()
        dali_iter = DALIGenericIterator(
            pipe, ["data", "label"], reader_name="Reader",
            last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True,
        )
        train_loader = DALIWrapper(dali_iter)

    if valid_size > 0:
        assert valid_batch > 0, "Validation batch size must be > 0"
        pipe = CifarPipeline(
            data_dir=valid_dir, batch_size=valid_batch, is_train=False,
            cutout_length=0, device_id=device_id, shard_id=rank,
            num_shards=world_size, num_workers=num_workers)
        pipe.build()
        dali_iter = DALIGenericIterator(
            pipe, ["data", "label"], reader_name="Reader",
            last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True,
        )
        valid_loader = DALIWrapper(dali_iter)

    if test_batch > 0:
        pipe = CifarPipeline(
            data_dir=test_dir, batch_size=test_batch, is_train=False,
            cutout_length=0, device_id=device_id, shard_id=rank,
            num_shards=world_size, num_workers=num_workers)
        pipe.build()
        dali_iter = DALIGenericIterator(
            pipe, ["data", "label"], reader_name="Reader",
            last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True,
        )
        test_loader = DALIWrapper(dali_iter)

    return test_loader, train_loader, valid_loader