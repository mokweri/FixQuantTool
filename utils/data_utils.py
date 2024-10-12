import os
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch


class UniformDataset(Dataset):
    """
    get random uniform samples with mean 0 and variance 1
    """

    def __init__(self, length, size, transform):
        self.length = length
        self.transform = transform
        self.size = size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # var[U(-128, 127)] = (127 - (-128))**2 / 12 = 5418.75
        sample = (torch.randint(high=255, size=self.size).float() - 127.5) / 5418.75
        return sample


def getRandomData(dataset='imagenet', batch_size=512):
    """
    get random sample dataloader
    dataset: name of the dataset
    batch_size: the batch size of random data
    Works for input size 224
    """
    if dataset == 'cifar10':
        size = (3, 32, 32)
        num_data = 10000
    elif dataset == 'imagenet':
        num_data = 10000
        size = (3, 224, 224)
    else:
        raise NotImplementedError

    dataset = UniformDataset(length=num_data, size=size, transform=None)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=32)
    return data_loader


def getValData(dataset='imagenet', batch_size=1024, path='data/imagenet', num_workers=32, for_inception=False):
    """
    Get dataloader of testset
    dataset: name of the dataset
    batch_size: the batch size of random data
    path: the path to the data
    for_inception: whether the data is for Inception because inception has input size 299 rather than 224
    """
    if dataset == 'imagenet':
        input_size = 299 if for_inception else 224
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        data_dir = os.path.join(path, 'val')
        val_dataset = datasets.ImageFolder(
            data_dir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True)

        return val_loader

    elif dataset == 'cifar10':
        data_dir = path
        normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                         std=(0.2023, 0.1994, 0.2010))
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        val_dataset = datasets.CIFAR10(root=data_dir,
                                       train=False,
                                       transform=transform_test)
        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers)
        return val_loader
    else:
        raise NotImplementedError


def getTrainData(dataset='imagenet', batch_size=512, path='data/imagenet', num_workers=32,
                 for_inception=False, download=False, data_percentage=1.0):
    """
    Get dataloader of training
    dataset: name of the dataset
    batch_size: the batch size of random data
    path: the path to the data
    for_inception: whether the data is for Inception because inception has input size 299 rather than 224
    """
    if dataset == 'imagenet':
        input_size = 299 if for_inception else 224
        data_dir = os.path.join(path, 'train')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            data_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        dataset_length = int(len(train_dataset) * data_percentage)
        partial_train_dataset, _ = torch.utils.data.random_split(train_dataset,
                                                                 [dataset_length, len(train_dataset) - dataset_length])

        train_loader = torch.utils.data.DataLoader(partial_train_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, pin_memory=True)

        return train_loader

    elif dataset == 'cifar10':
        data_dir = path
        normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                         std=(0.2023, 0.1994, 0.2010))
        transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                             transforms.RandomCrop(32, padding=4),
                                             transforms.ToTensor(),
                                             normalize])
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, transform=transform_train, download=download)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers)

        return train_loader
    else:
        raise NotImplementedError


if __name__ == '__main__':
    # Test usage
    data_ = getTrainData("cifar10", batch_size=24, num_workers=8, download=False, path="../../data")

