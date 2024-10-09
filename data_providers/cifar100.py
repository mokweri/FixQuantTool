import warnings
import os
import math
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from data_providers.base_provider import DataProvider
from data_providers.my_dataloader import MyRandomResizedCrop, MyDistributedSampler

__all__ = ["Cifar100DataProvider"]


class Cifar100DataProvider(DataProvider):
    DEFAULT_PATH = "./dataset/cifar100"

    def __init__(
            self,
            save_path=None,
            train_batch_size=256,
            test_batch_size=512,
            valid_size=None,
            n_worker=32,
            image_size=32,
            num_replicas=None,
            rank=None,
    ):

        warnings.filterwarnings("ignore")
        self._save_path = save_path

        self.image_size = image_size  # int or list of int

        self._valid_transform_dict = {}
        valid_transforms = self.build_valid_transform()
        train_loader_class = torch.utils.data.DataLoader
        train_dataset = self.train_dataset(self.build_train_transform())

        if valid_size is not None:
            if not isinstance(valid_size, int):
                assert isinstance(valid_size, float) and 0 < valid_size < 1
                valid_size = int(len(train_dataset) * valid_size)

            valid_dataset = self.train_dataset(valid_transforms)
            train_indexes, valid_indexes = self.random_sample_valid_set(
                len(train_dataset), valid_size
            )

            if num_replicas is not None:
                train_sampler = MyDistributedSampler(
                    train_dataset, num_replicas, rank, True, np.array(train_indexes)
                )
                valid_sampler = MyDistributedSampler(
                    valid_dataset, num_replicas, rank, True, np.array(valid_indexes)
                )
            else:
                train_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                    train_indexes
                )
                valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                    valid_indexes
                )

            self.train = train_loader_class(
                train_dataset,
                batch_size=train_batch_size,
                sampler=train_sampler,
                num_workers=n_worker,
                pin_memory=False,
            )
            self.valid = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=test_batch_size,
                sampler=valid_sampler,
                num_workers=n_worker,
                pin_memory=False,
            )
        else:
            if num_replicas is not None:
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_dataset, num_replicas, rank
                )
                self.train = train_loader_class(
                    train_dataset,
                    batch_size=train_batch_size,
                    sampler=train_sampler,
                    num_workers=n_worker,
                    pin_memory=True,
                )
            else:
                self.train = train_loader_class(
                    train_dataset,
                    batch_size=train_batch_size,
                    shuffle=True,
                    num_workers=n_worker,
                    pin_memory=False,
                )
            self.valid = None

        test_dataset = self.test_dataset(valid_transforms)
        if num_replicas is not None:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset, num_replicas, rank
            )
            self.test = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=test_batch_size,
                sampler=test_sampler,
                num_workers=n_worker,
                pin_memory=False,
            )
        else:
            self.test = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=test_batch_size,
                shuffle=True,
                num_workers=n_worker,
                pin_memory=False,
            )

        if self.valid is None:
            self.valid = self.test

    @staticmethod
    def name():
        return "cifar100"

    @property
    def data_shape(self):
        return 3, self.image_size, self.image_size  # C, H, W

    @property
    def n_classes(self):
        return 100

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = self.DEFAULT_PATH
            if not os.path.exists(self._save_path):
                self._save_path = os.path.expanduser("~/dataset/cifar100")
        return self._save_path

    @property
    def data_url(self):
        raise ValueError("unable to download %s" % self.name())

    def train_dataset(self, _transforms):
        return datasets.CIFAR100(self.train_path, train=True, transform=_transforms, download=True)

    def test_dataset(self, _transforms):
        return datasets.CIFAR100(self.valid_path, train=False, transform=_transforms, download=True)

    @property
    def train_path(self):
        return os.path.join(self.save_path, "train")

    @property
    def valid_path(self):
        return os.path.join(self.save_path, "val")

    @property
    def normalize(self):
        return transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    def build_train_transform(self, ):
        # random_resize_crop -> random_horizontal_flip
        train_transforms = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # AutoAugment(),
        ]

        train_transforms += [
            transforms.ToTensor(),
            # self.normalize,
        ]

        train_transforms = transforms.Compose(train_transforms)
        return train_transforms

    def build_valid_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            self.normalize,
        ])

    def build_sub_train_loader(
            self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None
    ):
        # used for resetting BN running statistics
        if self.__dict__.get("sub_train_%d" % self.image_size, None) is None:
            if num_worker is None:
                num_worker = self.train.num_workers

            n_samples = len(self.train.dataset)
            g = torch.Generator()
            g.manual_seed(DataProvider.SUB_SEED)
            rand_indexes = torch.randperm(n_samples, generator=g).tolist()

            new_train_dataset = self.train_dataset(
                self.build_train_transform(
                    image_size=self.image_size, print_log=False
                )
            )
            chosen_indexes = rand_indexes[:n_images]
            if num_replicas is not None:
                sub_sampler = MyDistributedSampler(
                    new_train_dataset,
                    num_replicas,
                    rank,
                    True,
                    np.array(chosen_indexes),
                )
            else:
                sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                    chosen_indexes
                )
            sub_data_loader = torch.utils.data.DataLoader(
                new_train_dataset,
                batch_size=batch_size,
                sampler=sub_sampler,
                num_workers=num_worker,
                pin_memory=False,
            )
            self.__dict__["sub_train_%d" % self.image_size] = []
            for images, labels in sub_data_loader:
                self.__dict__["sub_train_%d" % self.image_size].append(
                    (images, labels)
                )
        return self.__dict__["sub_train_%d" % self.image_size]
