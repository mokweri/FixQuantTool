from fixquant.utils import calc_learning_rate, build_optimizer
from fixquant.data import ImagenetDataProvider
from fixquant.data import Cifar10DataProvider
from fixquant.data import Cifar100DataProvider

import torch

__all__ = ["BaseConfig", "RunConfig", "DistributedRunConfig"]


class BaseConfig:
    def __init__(
        self,
        n_epochs,
        warmup_epochs,
        init_lr,
        lr_schedule_type,
        lr_schedule_param,
        is_qat,
        quantizer_lr,
        quantizer_lr_decay,
        dataset,
        train_batch_size,
        test_batch_size,
        valid_size,
        opt_type,
        opt_param,
        weight_decay,
        no_decay_keys,
        validation_frequency,
        print_frequency,
    ):
        self.n_epochs = n_epochs
        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
        self.lr_schedule_type = lr_schedule_type
        self.lr_schedule_param = lr_schedule_param

        self.is_qat = is_qat
        self.quantizer_lr = quantizer_lr
        self.quantizer_lr_decay = quantizer_lr_decay

        self.dataset = dataset
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.valid_size = valid_size

        self.opt_type = opt_type
        self.opt_param = opt_param
        self.weight_decay = weight_decay
        self.no_decay_keys = no_decay_keys

        self.validation_frequency = validation_frequency
        self.print_frequency = print_frequency
       
    @property
    def config(self):
        config = {}
        for key in self.__dict__:
            if not key.startswith("_"):
                config[key] = self.__dict__[key]
        return config

    """ learning rate """
    def adjust_learning_rate(self, optimizer, train_loader, epoch, batch_idx, ddp=False):
        if self.is_qat:
            """FOR TQT: Sets the learning rate to the initial LR decayed by decay ratios"""
            weight_lr_decay_steps = 3000 * (24 / self.train_batch_size)
            quantizer_lr_decay_steps = 1000 * (24 / self.train_batch_size)
            weight_lr_decay = 0.94

            step = len(train_loader) * epoch + batch_idx

            for param_group in optimizer.param_groups:
                group_name = param_group['name']
                if group_name == 'weight' and step % weight_lr_decay_steps == 0:
                    lr = self.init_lr * (weight_lr_decay ** (step / weight_lr_decay_steps))
                    param_group['lr'] = lr
                    print('{} lr at epoch {}, step {}:, lr={}'.format(group_name, epoch, step,lr))
                if group_name == 'quantizer' and step % quantizer_lr_decay_steps == 0:
                    lr = self.quantizer_lr * (
                            self.quantizer_lr_decay ** (step / quantizer_lr_decay_steps))
                    param_group['lr'] = lr
                    print('{} lr at epoch {}, step {}:, lr={}'.format(group_name, epoch, step,lr))

        else:
            """adjust learning of a given optimizer and return the new learning rate"""
            new_lr = calc_learning_rate(
                init_lr=self.init_lr,
                epoch=epoch,
                n_epochs=self.n_epochs,
                batch_idx=batch_idx,
                n_batch=len(train_loader),
                train_loader_length=len(train_loader),
                lr_schedule_type="cosine",
                ddp=ddp,  # Set to True if using DDP
                warmup_epochs=self.warmup_epochs,
                hvd_size=1,
                batches_per_allreduce=1
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr
            return new_lr

    """ data provider """
    @property
    def data_provider(self):
        raise NotImplementedError

    @property
    def train_loader(self):
        return self.data_provider.train_loader

    @property
    def val_loader(self):
        return self.data_provider.val_loader

    @property
    def test_loader(self):
        return self.data_provider.test_loader

    def random_sub_train_loader(
        self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None
    ):
        return self.data_provider.build_sub_train_loader(
            n_images, batch_size, num_worker, num_replicas, rank
        )

    """ optimizer """
    def build_optimizer(self, net_params):
        if self.is_qat:
            # optimizer for tqt
            return torch.optim.Adam(net_params, self.init_lr, weight_decay=self.weight_decay)
        else:
            return build_optimizer(
                net_params,
                self.opt_type,
                self.opt_param,
                self.init_lr,
                self.weight_decay,
                self.no_decay_keys,
            )


class RunConfig(BaseConfig):
    def __init__(
        self,
        n_epochs=150,
        warmup_epochs=5,
        init_lr=0.05,
        lr_schedule_type="cosine",
        lr_schedule_param=None,
        is_qat=False,
        quantizer_lr=1e-2,
        quantizer_lr_decay=0.5,
        dataset="imagenet", # 'cifar10' or 'cifar100'
        train_batch_size=32,
        test_batch_size=16,
        valid_size=None,
        opt_type="sgd",
        opt_param=None,
        weight_decay=4e-5,
        no_decay_keys=None,
        validation_frequency=1,
        print_frequency=10,
        n_worker=32,
        image_size=32, # 224, # 32
        **kwargs
    ):
        super(RunConfig, self).__init__(
            n_epochs,
            warmup_epochs,
            init_lr,
            lr_schedule_type,
            lr_schedule_param,
            is_qat,
            quantizer_lr,
            quantizer_lr_decay,
            dataset,
            train_batch_size,
            test_batch_size,
            valid_size,
            opt_type,
            opt_param,
            weight_decay,
            no_decay_keys,
            validation_frequency,
            print_frequency,
        )

        self.n_worker = n_worker
        self.image_size = image_size

    @property
    def data_provider(self):
        if self.__dict__.get("_data_provider", None) is None:
            if self.dataset == ImagenetDataProvider.name():
                DataProviderClass = ImagenetDataProvider
            elif self.dataset == Cifar10DataProvider.name():   
                DataProviderClass = Cifar10DataProvider
            elif self.dataset == Cifar100DataProvider.name():   
                DataProviderClass = Cifar100DataProvider        
            else:
                raise NotImplementedError
            self.__dict__["_data_provider"] = DataProviderClass(
                train_batch_size=self.train_batch_size,
                test_batch_size=self.test_batch_size,
                valid_size=self.valid_size,
                n_worker=self.n_worker,
                image_size=self.image_size,
                pin_memory=True
            )
        return self.__dict__["_data_provider"]

    def print_config(self):
        print("Run configurations:")
        for k, v in self.config.items():
            print("\t%s: %s" % (k, v))

class DistributedRunConfig(RunConfig):
    def __init__(
        self,
        n_epochs=150,
        warmup_epochs=5,
        init_lr=0.05,
        lr_schedule_type="cosine",
        lr_schedule_param=None,
        dataset="imagenet",
        train_batch_size=64,
        test_batch_size=64,
        valid_size=None,
        opt_type="sgd",
        opt_param=None,
        weight_decay=4e-5,
        label_smoothing=0.1,
        no_decay_keys=None,
        validation_frequency=1,
        print_frequency=10,
        n_worker=8,
        batches_per_allreduce=1,
        image_size=224,
        **kwargs
    ):
        super(DistributedRunConfig, self).__init__(
            n_epochs,
            warmup_epochs,
            init_lr,
            lr_schedule_type,
            lr_schedule_param,
            dataset,
            train_batch_size,
            test_batch_size,
            valid_size,
            opt_type,
            opt_param,
            weight_decay,
            label_smoothing,
            no_decay_keys,
            validation_frequency,
            print_frequency,
            n_worker,
            **kwargs
        )

        self._num_replicas = kwargs["num_replicas"]
        self._rank = kwargs["rank"]
        self.hvd_size = kwargs["hvd_size"]
        self.image_size = image_size
        self.batches_per_allreduce = batches_per_allreduce

    @property
    def data_provider(self):
        if self.__dict__.get("_data_provider", None) is None:
            if self.dataset == ImagenetDataProvider.name():
                DataProviderClass = ImagenetDataProvider
            elif self.dataset == Cifar10DataProvider.name():
                DataProviderClass = Cifar10DataProvider  
            elif self.dataset == Cifar100DataProvider.name():
                DataProviderClass = Cifar100DataProvider        
            else:
                raise NotImplementedError
            if self.dataset == "imagenet":
                self.__dict__["_data_provider"] = DataProviderClass(
                    train_batch_size=self.train_batch_size,
                    test_batch_size=self.test_batch_size,
                    valid_size=self.valid_size,
                    n_worker=self.n_worker,
                    image_size=self.image_size,
                    num_replicas=self._num_replicas,
                    rank=self._rank,
                    pin_memory=True
                )
            else:
                self.__dict__["_data_provider"] = DataProviderClass(
                    train_batch_size=self.train_batch_size,
                    test_batch_size=self.test_batch_size,
                    valid_size=self.valid_size,
                    n_worker=self.n_worker,
                    image_size=self.image_size,
                    num_replicas=self._num_replicas,
                    rank=self._rank,
                    pin_memory=True
                ) 
        return self.__dict__["_data_provider"]

    def adjust_learning_rate(self, optimizer, train_loader, epoch, batch_idx, ddp=True):
        new_lr = calc_learning_rate(
            init_lr=self.init_lr,
            epoch=epoch,
            n_epochs=self.n_epochs,
            batch_idx=batch_idx,
            n_batch=len(train_loader),
            train_loader_length=len(train_loader),
            lr_schedule_type=self.lr_schedule_type,
            ddp=ddp,  # Set to True if using DDP
            warmup_epochs=self.warmup_epochs,
            hvd_size=self.hvd_size,
            batches_per_allreduce=self.batches_per_allreduce
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr

        