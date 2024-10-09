import argparse
import os
import shutil
import time
import json

import torch
import torch.nn as nn
import torch.optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from model_transforms import create_quantizable_model, create_qconfig, standardize_qconfig

from utils.pytorch_utils2 import build_optimizer
from data_providers.imagenet import ImagenetDataProvider
from run_manager import ClassificationRunConfig, RunManager

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_dir',
    default=r"C:\Users\oma02\Downloads\ILSVRC\Data\CLS-LOC",
    help='Data set directory.')
parser.add_argument(
    '--workers',
    default=4,
    type=int,
    help='Number of data loading workers to be used.')
parser.add_argument('--epochs', default=3, type=int, help='Training epochs.')
parser.add_argument(
    '--weight_lr',
    default=1e-5,
    type=float,
    help='Initial learning rate of network weights.')
parser.add_argument(
    '--weight_lr_decay',
    default=0.94,
    type=int,
    help='Learning rate decay ratio of network weights.')
parser.add_argument(
    '--train_batch_size', default=24, type=int, help='Batch size for training.')
parser.add_argument(
    '--val_batch_size',
    default=100,
    type=int,
    help='Batch size for validation.')
parser.add_argument(
    '--weight_decay', default=1e-4, type=float, help='Weight decay.')
parser.add_argument(
    '--display_freq',
    default=100,
    type=int,
    help='Display training metrics every n steps.')
parser.add_argument(
    '--val_freq', default=1000, type=int, help='Validate model every n steps.')
parser.add_argument(
    '--mode',
    default='train',
    choices=['train', 'deploy'],
    help='Running mode.')
parser.add_argument(
    '--save_dir',
    default='./qat_models',
    help='Directory to save trained models.')
parser.add_argument(
    '--output_dir', default='qat_result', help='Directory to save qat result.')
parser.add_argument(
    '--gpus',
    type=str,
    default='0',
    help='gpu ids to be used for training, seperated by commas')
args, _ = parser.parse_known_args()


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


def mkdir_if_not_exist(x):
    if not x or os.path.isdir(x):
        return
    os.mkdir(x)
    if not os.path.isdir(x):
        raise RuntimeError("Failed to create dir %r" % x)


def save_checkpoint(state, is_best, directory):
    mkdir_if_not_exist(directory)

    filepath = os.path.join(directory, 'model.pth')
    torch.save(state, filepath)
    if is_best:
        best_acc1 = state['best_acc1'].item()
        best_filepath = os.path.join(directory, 'model_best_%5.3f.pth' % best_acc1)
        shutil.copyfile(filepath, best_filepath)
        print('Saving best ckpt to {}, acc1: {}'.format(best_filepath, best_acc1))
    return best_filepath if is_best else filepath


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, step):
    """Sets the learning rate to the initial LR decayed by decay ratios"""
    weight_lr_decay_steps = 3000 * (24 / args.train_batch_size)

    for param_group in optimizer.param_groups:
        new_lr = args.weight_lr * (args.weight_lr_decay ** (step / weight_lr_decay_steps))
        param_group['lr'] = new_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(model, train_loader, val_loader, criterion, device_ids):
    best_acc1 = 0
    best_filepath = None

    num_train_batches_per_epoch = int(len(train_loader) / args.train_batch_size)
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

    # optimizer
    net_params = []
    for param in model.parameters():
        if param.requires_grad:
            net_params.append(param)

    optimizer = build_optimizer(net_params, "adam", opt_param=None, init_lr=args.weight_lr,
                                weight_decay=args.weight_decay, no_decay_keys=None)

    for epoch in range(args.epochs):
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

                # remember best acc@1 and save checkpoint
                is_best = acc1 > best_acc1
                best_acc1 = max(acc1, best_acc1)

                filepath = save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict() if not isinstance(model, nn.DataParallel) \
                            else model.module.state_dict(),
                        'best_acc1': best_acc1
                    }, is_best, args.save_dir)
                if is_best:
                    best_filepath = filepath

    return best_filepath


def main():
    print('Used arguments:', args)

    traindir = os.path.join(args.data_dir, 'train')
    valdir = os.path.join(args.data_dir, 'val')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    device_ids = None if args.gpus == "" else [int(i) for i in args.gpus.split(",")]
    device = f"cuda:{device_ids[0]}" if device_ids is not None and len(device_ids) > 0 else "cpu"

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    #model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    # qconfig = create_qconfig(model, run_config.valid_loader, bitwidth=8)
    # qconfig = standardize_qconfig(qconfig)
    # # save the dict for ease of future use
    # with open('qconfig_vgg.json', 'w') as json_file:
    #     json.dump(qconfig, json_file)

    # load qconfig
    with open('qconfig.json', 'r') as json_file:
        qconfig = json.load(json_file)

    # for key, value in qconfig.items():
    #     print(f"{key}: {value}")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    inputs = torch.randn([args.train_batch_size, 3, 224, 224], dtype=torch.float32, device=device)

    if args.mode == 'train':
        # Step 1: Get quantized model and train it.
        quantized_model = create_quantizable_model(model, qconfig)
        criterion = criterion.to(device)
        best_ckpt = train(quantized_model, train_loader, val_loader, criterion, device_ids)

        validate(val_loader, quantized_model, criterion, device)

    elif args.mode == 'deploy':
        pass
    else:
        raise ValueError('mode must be one of ["train", "deploy"]')


if __name__ == '__main__':
    # with torch.autograd.set_detect_anomaly(True):
    main()
