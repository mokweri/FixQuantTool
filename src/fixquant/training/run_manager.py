import os
import random
import tempfile
import time
import json
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from tqdm import tqdm

from fixquant.utils import (
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
            # log_thresholds are already regularised by the explicit TQT norm
            # penalty in train_one_epoch; weight_decay on top of it would pull
            # every threshold toward 1.0 regardless of the data.
            net_params = [{
                'params': self.quantizer_parameters(self.net),
                'lr': self.run_config.quantizer_lr,
                'weight_decay': 0.0,
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

        self._q_params = self.quantizer_parameters(self.net)
        self._thresholds_frozen = False

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
    @staticmethod
    def _atomic_torch_save(checkpoint, path):
        """Write a checkpoint completely before replacing its destination."""
        directory = os.path.dirname(path)
        fd, tmp_path = tempfile.mkstemp(
            dir=directory,
            prefix=f".{os.path.basename(path)}.",
            suffix=".tmp",
        )
        os.close(fd)
        try:
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def save_checkpoint(self, checkpoint=None, is_best=False, model_name=None):
        if checkpoint is None:
            checkpoint = {"state_dict": self.network.state_dict()}
        checkpoint = dict(checkpoint)

        if model_name is None:
            model_name = "latest.pth.tar"

        checkpoint["dataset"] = self.run_config.dataset  # add `dataset` info to the checkpoint
        model_path = os.path.join(self.save_path, model_name)

        self._atomic_torch_save(checkpoint, model_path)
        if is_best:
            best_path = os.path.join(self.save_path, "model_best.pth.tar")
            self._atomic_torch_save(checkpoint, best_path)

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

                # TQT threshold norm penalty (Jain et al. 2019, sec. 4.2)
                if self.run_config.is_qat and not self._thresholds_frozen:
                    l2_decay = 1e-4
                    l2_norm = 0.0
                    for param in self._q_params:
                        l2_norm += torch.pow(param, 2.0)[0]
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

    def freeze_thresholds(self):
        """Stop training TQT thresholds so exported frac bits are stable."""
        from fixquant.quantization.tqt_quantizer import TQTQuantizer
        for m in self.network.modules():
            if isinstance(m, TQTQuantizer):
                m.freeze_quant(True)
        self._thresholds_frozen = True
        print("[RunManager] Quantizer thresholds frozen.")

    def log_quantizer_state(self, epoch):
        from fixquant.diagnostics import log_quantizer_state
        log_quantizer_state(self.network, epoch,
                            os.path.join(self.logs_path, "quant_thresholds.csv"))

    def train(self, warmup_epoch=0):
        n_epochs = self.run_config.n_epochs + warmup_epoch
        freeze_frac = getattr(self.run_config, "threshold_freeze_frac", None)
        freeze_epoch = int(n_epochs * freeze_frac) if (
            self.run_config.is_qat and freeze_frac is not None) else None

        for epoch in range(self.start_epoch, n_epochs):
            if freeze_epoch is not None and epoch >= freeze_epoch and not self._thresholds_frozen:
                self.freeze_thresholds()

            train_loss, train_top1, train_top5 = self.train_one_epoch(epoch)

            if self.run_config.is_qat:
                self.log_quantizer_state(epoch)

            validated = (epoch + 1) % self.run_config.validation_frequency == 0
            if validated:
                val_loss, val_acc, val_acc5 = self.validate( epoch=epoch, is_test=False)

                is_best = float(val_acc) > float(self.best_acc)
                self.best_acc = max(float(self.best_acc), float(val_acc))
            else:
                is_best = False

            checkpoint = {
                "epoch": epoch,
                "best_acc": self.best_acc,
                "train_loss": float(train_loss),
                "train_top1": float(train_top1),
                "train_top5": float(train_top5),
                "optimizer": self.optimizer.state_dict(),
                "state_dict": self.network.state_dict(),
            }
            if validated:
                checkpoint.update({
                    "val_loss": float(val_loss),
                    "val_top1": float(val_acc),
                    "val_top5": float(val_acc5),
                })
            self.save_checkpoint(checkpoint, is_best=is_best)

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
