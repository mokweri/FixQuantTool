import argparse
import os
import time
import json
import math

import torch
import torch.nn as nn
import torch.optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from model_transforms import create_quantizable_model, create_qconfig, standardize_qconfig
from models.resnet_cifar import *

from utils.pytorch_utils import build_optimizer, save_checkpoint
from utils.common_tools import *
from utils.data_utils import getTrainData, getValData

from data_providers.imagenet import ImagenetDataProvider
from run_manager import ClassificationRunConfig, RunManager

# Parse arguments
parser = argparse.ArgumentParser()

# Datasets
parser.add_argument('-d', '--data_dir',
                    default=r"C:\Users\oma02\Downloads\ILSVRC\Data\CLS-LOC", help='Data set directory.')
parser.add_argument('-j', '--workers',
                    default=8, type=int, help='Number of data loading workers to be used.')

# Optimization options
parser.add_argument('--epochs',
                    default=100, type=int, help='No. of training epochs.')
parser.add_argument('--start_epoch',
                    default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--warmup_epoch',
                    default=0, type=int, metavar='N', help='manual warmup epoch number (useful on restarts)')
parser.add_argument('--train_batch',
                    default=32, type=int, metavar='N', help='train batchsize (default: 32)')
parser.add_argument('--val_batch',
                    default=32, type=int, metavar='N', help='validation batchsize (default: 32)')
parser.add_argument('--lr', '--learning-rate',
                    default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--lr_type',
                    default='cos', type=str, help='lr scheduler (exp/cos/step3/fixed)')
parser.add_argument('--schedule',
                    type=int, nargs='+', default=[31, 61, 91], help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma',
                    type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum',
                    default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Quantization
parser.add_argument('--mode',
                    default='QAT', choices=['train', 'QAT', 'deploy'], help='Running mode.')
# Miscs
parser.add_argument('--display_freq',
                    default=100, type=int, help='Display training metrics every n steps.')
parser.add_argument('--val_freq',
                    default=100000, type=int, help='Validate model every n steps.')  # for imagenet increase it
parser.add_argument('--save_dir',
                    default='./qat_models', help='Directory to save trained models.')
parser.add_argument('--output_dir',
                    default='qat_result', help='Directory to save qat result.')
# Device options
parser.add_argument('--gpus',
                    type=str, default='0', help='gpu ids to be used for training, seperated by commas')

args, _ = parser.parse_known_args()

# Global variables for tracking the learning rate and accuracy
lr_current = None
best_acc = 0


def validate(val_loader, model, criterion, device):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    if not isinstance(model, nn.DataParallel):
        model = model.to(device)

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(
            top1=top1, top5=top5))

    return top1.avg


def adjust_learning_rate(optimizer, epoch, step):

    # Define the initial learning rate and set global variables for tracking
    global lr_current, best_acc

    initial_lr = args.lr if hasattr(args, 'lr') else 0.1  # Set default initial LR if not specified
    gamma = args.gamma if hasattr(args, 'gamma') else 0.1  # Set default gamma if not specified
    lr_type = args.lr_type if hasattr(args, 'lr_type') else 'cos'  # Default to cosine annealing if not specified

    if epoch < args.warmup_epoch:
        lr_current = initial_lr * (epoch + 1) / args.warmup_epoch
    else:
        if lr_type == 'cos':
            lr_current = 0.5 * initial_lr * (1 + math.cos(math.pi * epoch / args.epochs))

        elif lr_type == 'exp':
            step = 1  # Change step size if needed
            lr_current = initial_lr * (gamma ** (epoch // step))

        elif lr_type == 'step':
            # Step decay: LR is reduced by gamma at specific epoch intervals defined in args.schedule
            lr_current = initial_lr
            if epoch in args.schedule:
                lr_current *= gamma

        elif lr_type == 'linear':
            lr_current = initial_lr * (1 - epoch / args.epochs)

        else:
            # Default: Use cosine annealing if no lr_type matches
            lr_current = 0.5 * initial_lr * (1 + math.cos(math.pi * epoch / args.epochs))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_current


def train_one_step(model, inputs, criterion, optimizer, step, device):
    # switch to train mode
    model.train()

    images, target = inputs
    if not isinstance(model, nn.DataParallel):
        model = model.to(device)
    images = images.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    # compute output
    output = model(images)
    loss = criterion(output, target)

    # measure accuracy and record loss
    acc1, acc5 = accuracy(output, target, topk=(1, 5))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, acc1, acc5


def train(model, train_loader, val_loader, optimizer, criterion, device_ids, start_epoch):
    best_acc1 = 0
    best_filepath = None

    num_train_batches_per_epoch = int(len(train_loader) / args.train_batch)
    if device_ids is not None and len(device_ids) > 0:
        device = f"cuda:{device_ids[0]}"
        model = model.to(device)
        if len(device_ids) > 1:
            model = nn.DataParallel(model, device_ids=device_ids)

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    for epoch in range(start_epoch, args.epochs):
        progress = ProgressMeter(len(train_loader) * args.epochs, [batch_time, data_time, losses, top1, top5],
                                 prefix="Epoch[{}], Step: ".format(epoch))

        for i, (images, target) in enumerate(train_loader):
            end = time.time()
            # measure data loading time
            data_time.update(time.time() - end)

            step = len(train_loader) * epoch + i

            adjust_learning_rate(optimizer, epoch, step)
            loss, acc1, acc5 = train_one_step(model, (images, target), criterion, optimizer, step, device)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            if step % args.display_freq == 0:
                progress.display(step)

            if step % args.val_freq == 0:
                # evaluate on validation set
                acc1 = validate(val_loader, model, criterion, device)
                is_best = acc1 > best_acc1
                if is_best:
                    print('Saving..')
                    best_acc1 = acc1

                    filepath = save_checkpoint(
                        {
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict() if not isinstance(model, nn.DataParallel)
                            else model.module.state_dict(),
                            'best_acc1': best_acc1,
                            'optimizer': optimizer.state_dict(),
                        }, True, args.save_dir)
                if is_best:
                    best_filepath = filepath

    return best_filepath


def main():
    print('Used arguments:', args)

    # train_loader = getTrainData("imagenet", batch_size=args.train_batch, num_workers=8, path=args.data_dir)
    # val_loader = getValData("imagenet", batch_size=args.train_batch, num_workers=8, path=args.data_dir)

    data_dir = "/home/obed/Documents/Obed/data"
    train_loader = getTrainData("cifar10", batch_size=args.train_batch, num_workers=8, download=False, path=data_dir)
    val_loader = getValData("cifar10", batch_size=args.train_batch, num_workers=8, path=data_dir)

    device_ids = None if args.gpus == "" else [int(i) for i in args.gpus.split(",")]
    device = f"cuda:{device_ids[0]}" if device_ids is not None and len(device_ids) > 0 else "cpu"

    # model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    model = resnet18_cifar10()
    saved_filepath = './models/saved_models/resnet18_best_87.790.pth'
    checkpoint = torch.load(saved_filepath)
    model.load_state_dict(checkpoint['state_dict'])

    qconfig = create_qconfig(model, val_loader, bitwidth=8)
    qconfig = standardize_qconfig(qconfig)

    # # save the dict for ease of future use
    # with open('qconfig_vgg.json', 'w') as json_file:
    #     json.dump(qconfig, json_file)

    # load qconfig
    # with open('qconfig.json', 'r') as json_file:
    #     qconfig = json.load(json_file)

    # for key, value in qconfig.items():
    #     print(f"{key}: {value}")

    quantized_model = create_quantizable_model(model, qconfig)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    # optimizer
    net_params = []
    for param in model.parameters():
        if param.requires_grad:
            net_params.append(param)
    optimizer = build_optimizer(net_params, "sgd", opt_param=None, init_lr=args.lr,
                                weight_decay=args.weight_decay, no_decay_keys=None)

    # inputs = torch.randn([args.train_batch, 3, 224, 224], dtype=torch.float32, device=device)

    if args.mode == 'QAT':
        #best_ckpt = train(quantized_model, train_loader, val_loader, optimizer, criterion, device_ids)
        validate(val_loader, quantized_model, criterion, device)
    elif args.mode == 'deploy':
        pass
    else:
        raise ValueError('mode must be one of ["train", "QAT", "deploy"]')


if __name__ == '__main__':
    # with torch.autograd.set_detect_anomaly(True):
    main()
