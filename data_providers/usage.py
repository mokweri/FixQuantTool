from data_providers.imagenet import ImagenetDataProvider
from data_providers.cifar10 import Cifar10DataProvider
from data_providers.cifar100 import Cifar100DataProvider
from tqdm import tqdm

if __name__ == '__main__':
    # Cifar100DataProvider.DEFAULT_PATH = '/home/obed/Documents/CifarDatasets/cifar100'
    # data_provider = Cifar100DataProvider(
    #     train_batch_size=64,
    #     test_batch_size=64,
    #     n_worker=4,
    # )
    #
    # train_loader = data_provider.train_loader
    # val_loader = data_provider.val_loader
    #
    # with tqdm(total=len(val_loader), desc='Validate Loop:', disable=False) as t:
    #     for idx, (data, target) in enumerate(val_loader):
    #         print(data.shape)
    #         if idx == 100:
    #             break
    #         t.update(1)


    # Cifar10DataProvider.DEFAULT_PATH = '/home/obed/Documents/cifar_data'
    data_provider = Cifar10DataProvider(
        train_batch_size=62,
        test_batch_size=32,
        n_worker=4,
    )

    train_loader = data_provider.train_loader
    val_loader = data_provider.val_loader

    with tqdm(total=len(val_loader), desc='Validate Loop:', disable=False) as t:
        for idx, (data, target) in enumerate(val_loader):
            print(data.shape)
            if idx == 100:
                break
            t.update(1)


    # ImagenetDataProvider.DEFAULT_PATH = '/home/obed/Documents/imagenet'
    # data_provider = ImagenetDataProvider(
    #     train_batch_size=128,
    #     test_batch_size=62,
    #     image_size=224,
    #     pin_memory=True,
    # )
    # train_loader = data_provider.train_loader
    # val_loader = data_provider.val_loader
    #
    # calib_loader = data_provider.build_sub_train_loader(24,24)
    #
    # with tqdm(total=len(calib_loader), desc='Validate Loop:', disable=False) as t:
    #     for idx, (data, target) in enumerate(calib_loader):
    #         print(data.shape)
    #         # if idx == 1000:
    #         #     break
    #         t.update(1)
