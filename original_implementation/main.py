import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import args_file
import json
from utils import Logger
from models.build_model import build_model
from checkpoint_manager import CheckpointManager


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

best_acc1 = 0


def main():
    args = args_file.parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    # Checkpoint manager handles all weight saving/loading
    ckpt_manager = CheckpointManager(result_path=args.result_path, save_freq=args.save_freq)

    if not args.evaluate:
        if args.resume:
            with open(os.path.join(args.result_path, 'resume_args.json'), 'a') as f:
                json.dump(vars(args), f)
        else:
            with open(os.path.join(args.result_path, 'args.json'), 'w') as f:
                json.dump(vars(args), f)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    model = build_model(args)
    print(model)
    print(args)

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    if args.lr_scheduler == 'MultiStepLR':
        print("using MultiStepLR with steps: ", args.lr_steps)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, gamma=args.lr_reduce_factor)

    if args.resume:
        # Re-create scheduler first so load() can set last_epoch correctly
        if args.lr_scheduler == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, args.lr_steps, gamma=args.lr_reduce_factor)
        args.start_epoch, best_acc1 = ckpt_manager.load(
            args.resume, model, optimizer, scheduler, args)

    cudnn.benchmark = True

    # ---------------------------------------------------------------
    # CIFAR Data Loading
    # CIFAR-10 mean/std (change to CIFAR-100 values below if needed)
    # ---------------------------------------------------------------
    if args.dataset == 'cifar10':
        # CIFAR-10: 10 classes, 32x32 images
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
        train_dataset = datasets.CIFAR10(
            root=args.data, train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        )
        val_dataset = datasets.CIFAR10(
            root=args.data, train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
    elif args.dataset == 'cifar100':
        # CIFAR-100: 100 classes, 32x32 images
        normalize = transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
        train_dataset = datasets.CIFAR100(
            root=args.data, train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        )
        val_dataset = datasets.CIFAR100(
            root=args.data, train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
    else:
        raise ValueError("Unsupported dataset '{}'. Choose 'cifar10' or 'cifar100'.".format(args.dataset))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    if args.resume:
        mode = 'a'
    else:
        mode = 'w'

    t_log_name = "train" + args.dataset + ".log"
    v_log_name = "val" + args.dataset + ".log"
    train_logger = Logger(os.path.join(args.result_path, t_log_name),
                          ['epoch', 'loss', 'acc1', 'acc5', 'lr'], mode=mode)
    val_logger = Logger(os.path.join(args.result_path, v_log_name),
                        ['epoch', 'loss', 'acc1', 'acc5'], mode=mode)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if args.lr_scheduler == 'MultiStepLR':
            scheduler.step()
        else:
            adjust_learning_rate(optimizer, epoch, args)

        train_acc1, train_acc5, train_loss = \
            train(train_loader, model, criterion, optimizer, epoch, args)

        train_logger.log({
            'epoch': epoch + 1,
            'loss': '{:.4f}'.format(train_loss),
            'acc1': '{:.2f}'.format(train_acc1.item()),
            'acc5': '{:.2f}'.format(train_acc5.item()),
            'lr': '{:.6f}'.format(optimizer.param_groups[0]['lr'])
        })

        val_acc1, val_acc5, val_loss = validate(val_loader, model, criterion, args)

        val_logger.log({
            'epoch': epoch + 1,
            'loss': '{:.4f}'.format(val_loss),
            'acc1': '{:.2f}'.format(val_acc1.item()),
            'acc5': '{:.2f}'.format(val_acc5.item())
        })

        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            ckpt_manager.save(
                state={
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'val_acc1': val_acc1,
                    'optimizer': optimizer.state_dict(),
                },
                is_best=is_best,
                epoch=epoch + 1
            )


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch + 1))

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        output = model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)

    return top1.avg, top5.avg, losses.avg


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            output = model(input)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


class AverageMeter(object):
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
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
