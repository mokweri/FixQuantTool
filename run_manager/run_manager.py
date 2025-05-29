import os
import random
import time
import json
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from tqdm import tqdm

from utils import (
    AverageMeter,
    accuracy
)

__all__ = ["RunManager"]


class RunManager:
    def __init__(
        self, save_dir, net, run_config, no_gpu=False
    ):
        self.save_dir = save_dir
        self.net = net
        self.run_config = run_config

        self.best_acc = 0
        self.start_epoch = 0

        os.makedirs(self.save_dir, exist_ok=True)

        # move network to GPU if available
        if torch.cuda.is_available() and (not no_gpu):
            self.device = torch.device("cuda:0")
            self.net = self.net.to(self.device)
            cudnn.benchmark = True
        else:
            self.device = torch.device("cpu")

        # criterion
        self.train_criterion = nn.CrossEntropyLoss().to(self.device)
        self.test_criterion = nn.CrossEntropyLoss().to(self.device)

        # optimizer
        if self.run_config.is_qat:
            """ optimizer based on param groups - tqt """
            net_params = [{
                'params': self.quantizer_parameters(self.net),
                'lr': self.run_config.quantizer_lr,
                'name': 'quantizer'
            }, {
                'params': self.non_quantizer_parameters(self.net),
                'lr': self.run_config.init_lr,
                'name': 'weight'
            }]
        else:
            """ Normal training parameters """
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

    """ save path and log path """

    @property
    def save_path(self):
        if self.__dict__.get("_save_path", None) is None:
            save_path = os.path.join(self.save_dir, "checkpoint")
            os.makedirs(save_path, exist_ok=True)
            self.__dict__["_save_path"] = save_path
        return self.__dict__["_save_path"]

    @property
    def logs_path(self):
        if self.__dict__.get("_logs_path", None) is None:
            logs_path = os.path.join(self.save_dir, "logs")
            os.makedirs(logs_path, exist_ok=True)
            self.__dict__["_logs_path"] = logs_path
        return self.__dict__["_logs_path"]

    @property
    def network(self):
        return self.net.module if isinstance(self.net, nn.DataParallel) else self.net

    """ save and load models """

    def save_checkpoint(self, checkpoint=None, is_best=False, model_name=None):
        if checkpoint is None:
            checkpoint = {"state_dict": self.network.state_dict()}

        if model_name is None:
            model_name = "model_best.pth.tar"

        checkpoint["dataset"] = self.run_config.dataset  # add `dataset` info to the checkpoint
        model_path = os.path.join(self.save_path, model_name)

        torch.save(checkpoint, model_path)
        if is_best:
            best_path = os.path.join(self.save_path, "model_best.pth.tar")
            torch.save({"state_dict": checkpoint["state_dict"]}, best_path)

    """ metric related """
    def get_metric_dict(self):
        return {
            "top1": AverageMeter(),
            "top5": AverageMeter(),
        }

    """ train and test """
    def validate(self, epoch=0, is_test=False, run_str="", net=None, data_loader=None, no_logs=False,
        train_mode=False,
    ):
        if net is None:
            net = self.net

        if data_loader is None:
            data_loader = (
                self.run_config.test_loader if is_test else self.run_config.val_loader
            )
        if train_mode:
            net.train()
        else:
            net.eval()

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()


        with tqdm(total=len(data_loader), desc="Validate Epoch #{} {}".format(epoch + 1, run_str),
                  disable=no_logs, ) as t:
            with torch.no_grad():
                for i, (images, labels) in enumerate(data_loader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    # compute output
                    output = net(images)
                    loss = self.test_criterion(output, labels)

                    # measure accuracy and record loss
                    acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                    losses.update(loss.item(), images.size(0))
                    top1.update(acc1[0], images.size(0))
                    top5.update(acc5[0], images.size(0))

                    t.set_postfix({
                        'loss': float(losses.avg) if isinstance(losses.avg, torch.Tensor) else losses.avg,
                        'Acc@1': float(top1.avg) if isinstance(top1.avg, torch.Tensor) else top1.avg,
                        'Acc@5': float(top5.avg) if isinstance(top5.avg, torch.Tensor) else top5.avg
                    })
                    t.update(1)
        return losses.avg, top1.avg, top5.avg

  
    def train_one_epoch(self, epoch,):
        # switch to train mode
        self.net.train()

        nBatch = len(self.run_config.train_loader)

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        data_time = AverageMeter()

        with tqdm(
            total=nBatch,
            desc="{} Train Epoch #{}".format(self.run_config.dataset, epoch + 1),
        ) as t:
            end = time.time()
            for i, (images, labels) in enumerate(self.run_config.train_loader):
                data_time.update(time.time() - end)

                # new_lr = self.run_config.adjust_learning_rate(self.optimizer, self.run_config.train_loader, epoch, i, ddp=False)
                self.run_config.adjust_learning_rate(self.optimizer, self.run_config.train_loader, epoch, i, ddp=False)

                images, labels = images.to(self.device), labels.to(self.device)

                # compute output
                output = self.net(images)
                loss = self.train_criterion(output, labels)

                # tqt quantizer stuff
                if self.run_config.is_qat:
                    l2_decay = 1e-4
                    l2_norm = 0.0
                    quantizer_norm = True
                    q_params = self.quantizer_parameters(self.net)
                    for param in q_params:
                        l2_norm += torch.pow(param, 2.0)[0]
                    if quantizer_norm:
                        loss += l2_decay * torch.sqrt(l2_norm)

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure accuracy and record loss
                losses.update(loss.item(), images.size(0))
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                top1.update(acc1[0])
                top5.update(acc5[0])

                t.set_postfix(
                    {
                        "loss": losses.avg,
                        'Acc@1': top1.avg.item(),
                        'Acc@5': top5.avg.item(),
                        # "lr": new_lr,
                        "data_time": data_time.avg,
                    }
                )
                t.update(1)
                end = time.time()
        return losses.avg, top1.avg, top5.avg

    def train(self, warmup_epoch=0):

        for epoch in range(self.start_epoch, self.run_config.n_epochs + warmup_epoch):
            train_loss, train_top1, train_top5 = self.train_one_epoch(epoch)

            if (epoch + 1) % self.run_config.validation_frequency == 0:
                val_loss, val_acc, val_acc5 = self.validate( epoch=epoch, is_test=False)

                is_best = val_acc > self.best_acc
                self.best_acc = max(self.best_acc, val_acc)
            else:
                is_best = False

            self.save_checkpoint(
                {
                    "epoch": epoch,
                    "best_acc": self.best_acc,
                    "optimizer": self.optimizer.state_dict(),
                    "state_dict": self.network.state_dict(),
                },
                is_best=is_best,
            )

    @staticmethod
    def quantizer_parameters(model):
        return [
            param for name, param in model.named_parameters()
            if 'log_threshold' in name
        ]
    @staticmethod
    def non_quantizer_parameters(model):
        return [
            param for name, param in model.named_parameters()
            if 'log_threshold' not in name
        ]
