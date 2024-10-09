import os
import time
import json
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import math
from tqdm import tqdm

from utils import (
    AverageMeter,
    accuracy,
    write_log
)

__all__ = ["RunManager"]


def init_models(net, model_init="he_fout"):
    """
    Conv2d,
    BatchNorm2d, BatchNorm1d, GroupNorm
    Linear,
    """
    if isinstance(net, list):
        for sub_net in net:
            init_models(sub_net, model_init)
        return
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            if model_init == "he_fout":
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif model_init == "he_fin":
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            else:
                raise NotImplementedError
            if m.bias is not None:
                m.bias.data.zero_()
        elif type(m) in [nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm]:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            stdv = 1.0 / math.sqrt(m.weight.size(1))
            m.weight.data.uniform_(-stdv, stdv)
            if m.bias is not None:
                m.bias.data.zero_()


class RunManager:
    def __init__(
            self, path, net, run_config, init=True, measure_latency=None, no_gpu=False
    ):
        self.path = path
        self.net = net
        self.run_config = run_config

        self.best_acc = 0
        self.start_epoch = 0

        os.makedirs(self.path, exist_ok=True)

        # move network to GPU if available
        if torch.cuda.is_available() and (not no_gpu):
            self.device = torch.device("cuda:0")
            self.net = self.net.to(self.device)
            cudnn.benchmark = True
        else:
            self.device = torch.device("cpu")
        # initialize model (default)
        if init:
            init_models(run_config.model_init)

        # criterion
        self.train_criterion = nn.CrossEntropyLoss()
        self.test_criterion = nn.CrossEntropyLoss()

        # optimizer
        if self.run_config.no_decay_keys:
            keys = self.run_config.no_decay_keys.split("#")
            net_params = [
                self.network.get_parameters(
                    keys, mode="exclude"
                ),  # parameters with weight decay
                self.network.get_parameters(
                    keys, mode="include"
                ),  # parameters without weight decay
            ]
        else:
            # noinspection PyBroadException
            try:
                net_params = self.network.weight_parameters()
            except Exception:
                net_params = []
                for param in self.network.parameters():
                    if param.requires_grad:
                        net_params.append(param)
        self.optimizer = self.run_config.build_optimizer(net_params)
        self.net = torch.nn.DataParallel(self.net)

    """ save path and log path """

    @property
    def save_path(self):
        if self.__dict__.get("_save_path", None) is None:
            save_path = os.path.join(self.path, "checkpoint")
            os.makedirs(save_path, exist_ok=True)
            self.__dict__["_save_path"] = save_path
        return self.__dict__["_save_path"]

    @property
    def logs_path(self):
        if self.__dict__.get("_logs_path", None) is None:
            logs_path = os.path.join(self.path, "logs")
            os.makedirs(logs_path, exist_ok=True)
            self.__dict__["_logs_path"] = logs_path
        return self.__dict__["_logs_path"]

    @property
    def network(self):
        return self.net.module if isinstance(self.net, nn.DataParallel) else self.net

    def write_log(self, log_str, prefix="valid", should_print=True, mode="a"):
        write_log(self.logs_path, log_str, prefix, should_print, mode)

    """ save and load models """

    def save_model(self, checkpoint=None, is_best=False, model_name=None):
        if checkpoint is None:
            checkpoint = {"state_dict": self.network.state_dict()}

        if model_name is None:
            model_name = "checkpoint.pth.tar"

        checkpoint[
            "dataset"
        ] = self.run_config.dataset  # add `dataset` info to the checkpoint
        latest_fname = os.path.join(self.save_path, "latest.txt")
        model_path = os.path.join(self.save_path, model_name)
        with open(latest_fname, "w") as fout:
            fout.write(model_path + "\n")
        torch.save(checkpoint, model_path)

        if is_best:
            best_path = os.path.join(self.save_path, "model_best.pth.tar")
            torch.save({"state_dict": checkpoint["state_dict"]}, best_path)

    def load_model(self, model_fname=None):
        latest_fname = os.path.join(self.save_path, "latest.txt")
        if model_fname is None and os.path.exists(latest_fname):
            with open(latest_fname, "r") as fin:
                model_fname = fin.readline()
                if model_fname[-1] == "\n":
                    model_fname = model_fname[:-1]
        # noinspection PyBroadException
        try:
            if model_fname is None or not os.path.exists(model_fname):
                model_fname = "%s/checkpoint.pth.tar" % self.save_path
                with open(latest_fname, "w") as fout:
                    fout.write(model_fname + "\n")
            print("=> loading checkpoint '{}'".format(model_fname))
            checkpoint = torch.load(model_fname, map_location="cpu")
        except Exception:
            print("fail to load checkpoint from %s" % self.save_path)
            return {}

        self.network.load_state_dict(checkpoint["state_dict"])
        if "epoch" in checkpoint:
            self.start_epoch = checkpoint["epoch"] + 1
        if "best_acc" in checkpoint:
            self.best_acc = checkpoint["best_acc"]
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        print("=> loaded checkpoint '{}'".format(model_fname))
        return checkpoint

    def save_config(self, extra_run_config=None, extra_net_config=None):
        """dump run_config and net_config to the model_folder"""
        run_save_path = os.path.join(self.path, "run.config")
        if not os.path.isfile(run_save_path):
            run_config = self.run_config.config
            if extra_run_config is not None:
                run_config.update(extra_run_config)
            json.dump(run_config, open(run_save_path, "w"), indent=4)
            print("Run configs dump to %s" % run_save_path)

        try:
            net_save_path = os.path.join(self.path, "net.config")
            net_config = self.network.config
            if extra_net_config is not None:
                net_config.update(extra_net_config)
            json.dump(net_config, open(net_save_path, "w"), indent=4)
            print("Network configs dump to %s" % net_save_path)
        except Exception:
            print("%s do not support net config" % type(self.network))

    """ metric related """

    def get_metric_dict(self):
        return {
            "top1": AverageMeter(),
            "top5": AverageMeter(),
        }

    def update_metric(self, metric_dict, output, labels):
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        metric_dict["top1"].update(acc1[0].item(), output.size(0))
        metric_dict["top5"].update(acc5[0].item(), output.size(0))

    def get_metric_vals(self, metric_dict, return_dict=False):
        if return_dict:
            return {key: metric_dict[key].avg for key in metric_dict}
        else:
            return [metric_dict[key].avg for key in metric_dict]

    def get_metric_names(self):
        return "top1", "top5"

    """ train and test """

    def validate(
            self,
            epoch=0,
            is_test=False,
            run_str="",
            net=None,
            data_loader=None,
            no_logs=False,
            train_mode=False,
            extra_preprocess=None,
            att=None
    ):
        if net is None:
            net = self.net
        # if not isinstance(net, nn.DataParallel):
        #     net = nn.DataParallel(net)

        if data_loader is None:
            data_loader = (
                self.run_config.test_loader if is_test else self.run_config.valid_loader
            )

        if train_mode:
            net.train()
        else:
            net.eval()

        losses = AverageMeter()
        metric_dict = self.get_metric_dict()

        with torch.no_grad():
            with tqdm(
                    total=len(data_loader),
                    desc="Validate Epoch #{} {}".format(epoch + 1, run_str),
                    disable=no_logs,
            ) as t:
                for i, (images, labels) in enumerate(data_loader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    if extra_preprocess is not None:
                        for preprocess in extra_preprocess:
                            if att != None:
                                images = preprocess(images, att[0], att[1], att[2])
                            else:
                                images = preprocess(images)

                                # compute output
                    output = net(images)
                    loss = self.test_criterion(output, labels)
                    # measure accuracy and record loss
                    self.update_metric(metric_dict, output, labels)

                    losses.update(loss.item(), images.size(0))
                    t.set_postfix(
                        {
                            "loss": losses.avg,
                            **self.get_metric_vals(metric_dict, return_dict=True),
                            "img_size": images.size(2),
                        }
                    )
                    t.update(1)
        return losses.avg, self.get_metric_vals(metric_dict), output

    def train_one_epoch(self, args, epoch, warmup_epochs=0, warmup_lr=0):
        # switch to train mode
        self.net.train()

        nBatch = len(self.run_config.train_loader)

        losses = AverageMeter()
        metric_dict = self.get_metric_dict()
        data_time = AverageMeter()

        with tqdm(
                total=nBatch,
                desc="{} Train Epoch #{}".format(self.run_config.dataset, epoch + 1),
        ) as t:
            end = time.time()
            for i, (images, labels) in enumerate(self.run_config.train_loader):
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

                images, labels = images.to(self.device), labels.to(self.device)
                target = labels

                # compute output
                output = self.net(images)
                loss = self.train_criterion(output, labels)

                loss_type = "ce"

                # compute gradient and do SGD step
                self.net.zero_grad()  # or self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure accuracy and record loss
                losses.update(loss.item(), images.size(0))
                self.update_metric(metric_dict, output, target)

                t.set_postfix(
                    {
                        "loss": losses.avg,
                        **self.get_metric_vals(metric_dict, return_dict=True),
                        "img_size": images.size(2),
                        "lr": new_lr,
                        "loss_type": loss_type,
                        "data_time": data_time.avg,
                    }
                )
                t.update(1)
                end = time.time()
        return losses.avg, self.get_metric_vals(metric_dict)

    def train(self, args, warmup_epoch=0, warmup_lr=0):
        for epoch in range(self.start_epoch, self.run_config.n_epochs + warmup_epoch):
            train_loss, (train_top1, train_top5) = self.train_one_epoch(
                args, epoch, warmup_epoch, warmup_lr
            )

            if (epoch + 1) % self.run_config.validation_frequency == 0:
                val_loss, val_acc, val_acc5 = self.validate(
                    epoch=epoch, is_test=False
                )

                is_best = np.mean(val_acc) > self.best_acc
                self.best_acc = max(self.best_acc, np.mean(val_acc))
                val_log = "Valid [{0}/{1}]\tloss {2:.3f}\t{5} {3:.3f} ({4:.3f})".format(
                    epoch + 1 - warmup_epoch,
                    self.run_config.n_epochs,
                    np.mean(val_loss),
                    np.mean(val_acc),
                    self.best_acc,
                    self.get_metric_names()[0],
                )
                val_log += "\t{2} {0:.3f}\tTrain {1} {top1:.3f}\tloss {train_loss:.3f}\t".format(
                    np.mean(val_acc5),
                    *self.get_metric_names(),
                    top1=train_top1,
                    train_loss=train_loss
                )
                for v_a in val_acc:
                    val_log += "(%.3f), " % v_a
                self.write_log(val_log, prefix="valid", should_print=False)
            else:
                is_best = False

            self.save_model(
                {
                    "epoch": epoch,
                    "best_acc": self.best_acc,
                    "optimizer": self.optimizer.state_dict(),
                    "state_dict": self.network.state_dict(),
                },
                is_best=is_best,
            )
