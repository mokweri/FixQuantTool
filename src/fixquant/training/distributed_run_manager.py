import os
import json
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from fixquant.utils import (
    write_log,
)
from fixquant.utils import (
    DistributedMetric,
    list_mean,
    accuracy,
    AverageMeter,
)

__all__ = ["DistributedRunManager"]

class DistributedRunManager:
    def __init__(
        self,
        path,
        net,
        run_config,
        hvd_compression,
        backward_steps=1,
        is_root=False,
        init=True,
    ):
        import horovod.torch as hvd

        self.path = path
        self.net = net
        self.run_config = run_config
        self.is_root = is_root

        self.best_acc = 0.0
        self.start_epoch = 0

        os.makedirs(self.path, exist_ok=True)

        self.net.cuda()
        cudnn.benchmark = True


        # criterion
        self.train_criterion = nn.CrossEntropyLoss()
        self.test_criterion = nn.CrossEntropyLoss()

        # optimizer
        if self.run_config.no_decay_keys:
            keys = self.run_config.no_decay_keys.split("#")
            no_decay_params = []
            decay_params = []
            for name, param in self.net.named_parameters():
                if any(key in name for key in keys):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
            net_params = [decay_params, no_decay_params]
        else:
            try:
                net_params = self.network.weight_parameters()
            except Exception:
                net_params = []
                for param in self.network.parameters():
                    if param.requires_grad:
                        net_params.append(param)

        self.optimizer = self.run_config.build_optimizer(net_params)
        self.optimizer = hvd.DistributedOptimizer(
            self.optimizer,
            named_parameters=self.net.named_parameters(),
            compression=hvd_compression,
            backward_passes_per_step=backward_steps,
        )

    """ save path and log path """

    @property
    def save_path(self):
        if self.__dict__.get("_save_path", None) is None:
            save_path = os.path.join(self.path, "checkpoint")
            os.makedirs(save_path, exist_ok=True)
            self.__dict__["_save_path"] = save_path
        return self.__dict__["_save_path"]

    @property
    def network(self):
        return self.net

    @network.setter
    def network(self, new_val):
        self.net = new_val

    """ save & load model & save_config & broadcast """

    # noinspection PyArgumentList
    def broadcast(self):
        import horovod.torch as hvd

        self.start_epoch = hvd.broadcast(
            torch.LongTensor(1).fill_(self.start_epoch)[0], 0, name="start_epoch"
        ).item()
        self.best_acc = hvd.broadcast(
            torch.Tensor(1).fill_(self.best_acc)[0], 0, name="best_acc"
        ).item()
        hvd.broadcast_parameters(self.net.state_dict(), 0)
        hvd.broadcast_optimizer_state(self.optimizer, 0)


    """ metric related """
    def get_metric_dict(self):
        return {
            "top1": DistributedMetric("top1"),
            "top5": DistributedMetric("top5"),
        }

    def update_metric(self, metric_dict, output, labels):
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        metric_dict["top1"].update(acc1[0], output.size(0))
        metric_dict["top5"].update(acc5[0], output.size(0))

    def get_metric_vals(self, metric_dict, return_dict=False):
        if return_dict:
            return {key: metric_dict[key].avg.item() for key in metric_dict}
        else:
            return [metric_dict[key].avg.item() for key in metric_dict]

    def get_metric_names(self):
        return "top1", "top5"


    """ train & validate """
    def validate(
        self,
        epoch=0,
        is_test=False,
        run_str="",
        net=None,
        data_loader=None,
        no_logs=False,
    ):
        if net is None:
            net = self.net
        if data_loader is None:
            if is_test:
                data_loader = self.run_config.test_loader
            else:
                data_loader = self.run_config.val_loader

        net.eval()

        losses = DistributedMetric("val_loss")
        metric_dict = self.get_metric_dict()

        with torch.no_grad():
            with tqdm(
                total=len(data_loader),
                desc="Validate Epoch #{} {}".format(epoch + 1, run_str),
                disable=no_logs or not self.is_root,
            ) as t:
                for i, (images, labels) in enumerate(data_loader):
                    images, labels = images.cuda(), labels.cuda()
                    # compute output
                    output = net(images)
                    loss = self.test_criterion(output, labels)
                    # measure accuracy and record loss
                    losses.update(loss, images.size(0))
                    self.update_metric(metric_dict, output, labels)
                    t.set_postfix(
                        {
                            "loss": losses.avg.item(),
                            **self.get_metric_vals(metric_dict, return_dict=True),
                            "img_size": images.size(2),
                        }
                    )
                    t.update(1)
        return losses.avg.item(), self.get_metric_vals(metric_dict)

    

    def train_one_epoch(self, args, epoch, warmup_epochs=5, warmup_lr=0):
        self.net.train()
        self.run_config.train_loader.sampler.set_epoch(
            epoch
        )  # required by distributed sampler

        nBatch = len(self.run_config.train_loader)

        losses = DistributedMetric("train_loss")
        metric_dict = self.get_metric_dict()
        data_time = AverageMeter()

        with tqdm(
            total=nBatch,
            desc="Train Epoch #{}".format(epoch + 1),
            disable=not self.is_root,
        ) as t:
            end = time.time()
            for i, (images, labels) in enumerate(self.run_config.train_loader):
                MyRandomResizedCrop.BATCH = i
                data_time.update(time.time() - end)
                if epoch < warmup_epochs:
                    new_lr = self.run_config.warmup_adjust_learning_rate(
                        self.optimizer,
                        warmup_epochs * nBatch,
                        nBatch,
                        epoch,
                        i,
                        warmup_lr,
                    )
                else:
                    new_lr = self.run_config.adjust_learning_rate(
                        self.optimizer, epoch - warmup_epochs, i, nBatch
                    )

                images, labels = images.cuda(), labels.cuda()
                target = labels

                # compute output
                output = self.net(images)

                loss = self.train_criterion(output, labels)
                loss_type = "ce"

                # update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure accuracy and record loss
                losses.update(loss, images.size(0))
                self.update_metric(metric_dict, output, target)

                t.set_postfix(
                    {
                        "loss": losses.avg.item(),
                        **self.get_metric_vals(metric_dict, return_dict=True),
                        "img_size": images.size(2),
                        "lr": new_lr,
                        "loss_type": loss_type,
                        "data_time": data_time.avg,
                    }
                )
                t.update(1)
                end = time.time()

        return losses.avg.item(), self.get_metric_vals(metric_dict)

    def train(self, args, warmup_epochs=5, warmup_lr=0):
        for epoch in range(self.start_epoch, self.run_config.n_epochs + warmup_epochs):
            
            train_loss, (train_top1, train_top5) = self.train_one_epoch(
                args, epoch, warmup_epochs, warmup_lr
            )
            val_loss, val_top1, val_top5 = self.validate(epoch, is_test=False )

            is_best = list_mean(val_top1) > self.best_acc
            self.best_acc = max(self.best_acc, list_mean(val_top1))
            if self.is_root:
                val_log = (
                    "[{0}/{1}]\tloss {2:.3f}\t{6} acc {3:.3f} ({4:.3f})\t{7} acc {5:.3f}\t"
                    "Train {6} {top1:.3f}\tloss {train_loss:.3f}\t".format(
                        epoch + 1 - warmup_epochs,
                        self.run_config.n_epochs,
                        list_mean(val_loss),
                        list_mean(val_top1),
                        self.best_acc,
                        list_mean(val_top5),
                        *self.get_metric_names(),
                        top1=train_top1,
                        train_loss=train_loss
                    )
                )
                for v_a in val_top1:
                    val_log += "(%d, %.3f), " %  v_a
                self.write_log(val_log, prefix="valid", should_print=False)

                self.save_model(
                    {
                        "epoch": epoch,
                        "best_acc": self.best_acc,
                        "optimizer": self.optimizer.state_dict(),
                        "state_dict": self.net.state_dict(),
                    },
                    is_best=is_best,
                )
