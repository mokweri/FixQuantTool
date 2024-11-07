import torch
import torch.nn as nn
from utils.data_utils import getTrainData, getValData
import models.resnet_cifar as cifar_models
from utils.common_tools import *
from utils.pytorch_utils import save_checkpoint

import os
import time
import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--resume', '-r',
                    default=True, type=bool, help='Resume from checkpoint.')

# Optimization options
parser.add_argument('--epochs',
                    default=150, type=int, help='No. of training epochs.')
parser.add_argument('--train_batch',
                    default=24, type=int, metavar='N', help='train batchsize (default: 32)')
parser.add_argument('--lr',
                    default=0.1, type=float, help='learning rate')
# Miscs
parser.add_argument('--display_freq',
                    default=100, type=int, help='Display training metrics every n steps.')
parser.add_argument('--save_dir',
                    default='./saved_models-FP', help='Directory to save trained models.')
# Device options
parser.add_argument('--gpus',
                    type=str, default='0', help='gpu ids to be used for training, seperated by commas')

args = parser.parse_args()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


def train_one_step(model, inputs, criterion, optimizer, step, device):
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


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device_ids, start_epoch):
    global best_acc
    best_filepath = None

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
            # Measure data loading time
            data_time.update(time.time() - end)

            step = len(train_loader) * epoch + i

            loss, acc1, acc5 = train_one_step(model, (images, target), criterion, optimizer, step, device)

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            if step % args.display_freq == 0:
                progress.display(step)

        acc1 = validate(val_loader, model, criterion, device)
        is_best = acc1 > best_acc
        if is_best:
            print('Saving..')
            best_acc = acc1

            filepath = save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict() if not isinstance(model, nn.DataParallel)
                    else model.module.state_dict(),
                    'best_acc1': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, True, args.save_dir)
        if is_best:
            best_filepath = filepath

        scheduler.step()

    return best_filepath


if __name__ == '__main__':
    data_dir = r"C:\Users\oma02\OneDrive - Mälardalens universitet\Documents\Obed Workspaces\Python Projects\data"

    device_ids = None if args.gpus == "" else [int(i) for i in args.gpus.split(",")]
    device = f"cuda:{device_ids[0]}" if device_ids is not None and len(device_ids) > 0 else "cpu"

    train_loader = getTrainData("cifar10", batch_size=args.train_batch, num_workers=8, download=False, path=data_dir)
    val_loader = getValData("cifar10", batch_size=args.train_batch, num_workers=8, path=data_dir)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model = cifar_models.resnet50_cifar10()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    criterion = nn.CrossEntropyLoss().to(device)

    # if args.resume:
    #     # Load checkpoint.
    #     print('==> Resuming from checkpoint..')
    #     assert os.path.isdir('./saved_models-FP'), 'Error: no checkpoint directory found!'
    #     resume_filepath = './saved_models-FP/model_best_84.470.pth'
    #     checkpoint = torch.load(resume_filepath)
    #     model.load_state_dict(checkpoint['state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     scheduler.load_state_dict(checkpoint['scheduler'])
    #     best_acc = checkpoint['best_acc1']
    #     start_epoch = checkpoint['epoch']
    #
    # print(best_acc)
    # print(start_epoch)

    train(model, train_loader, val_loader, criterion, optimizer, scheduler, device_ids, start_epoch)
    validate(val_loader, model, criterion, device)

