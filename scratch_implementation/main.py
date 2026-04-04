"""
Usage:
    # Train iResNet-50 on CIFAR-10
    python iresnet_cifar.py --dataset cifar10

    # Train iResNet-50 on CIFAR-100
    python iresnet_cifar.py --dataset cifar100

    # Validate only from a checkpoint
    python iresnet_cifar.py --dataset cifar10 --evaluate --resume checkpoint.pth

    # Custom log / plot directories
    python iresnet_cifar.py --dataset cifar10 --log-dir ./logs --plot-dir ./plots
"""

import argparse
import logging
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from utils import *
from blocks import *
from iresnet import *
from get_data import *
from metrics import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=5e-4, type=float)
    parser.add_argument('--out_dir', default='Results', type=str)
    parser.add_argument('--log_path', default='Results/cifar10.log', type=str)
    args = parser.parse_args()

    # Configure Logger
    logging.basicConfig(
        filename=args.log_path,
        format='%(asctime)s %(levelname)s: %(message)s',
        filemode='w',
        level=logging.INFO,
        force=True
    )
    logger = logging.getLogger()

    ckpt_name = os.path.join(args.out_dir, f"{args.dataset}_checkpoint.pth")
    best_name = os.path.join(args.out_dir, f"{args.dataset}_model_best.pth")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data Loading
    start_data_time = time.time()
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), 
            (0.2023, 0.1994, 0.2010)
        ),
    ])

    if args.dataset == 'cifar10':
        datasets = torchvision.datasets.CIFAR10
        n_cls = 10
    else:
        datasets = torchvision.datasets.CIFAR100
        n_cls = 100

    trainset = datasets(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform_train
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2
    )

    testset = datasets(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform_train
    )

    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=100, 
        shuffle=False, 
        num_workers=2
    )

    data_load_time = time.time() - start_data_time
    model = iresnet([3, 4, 6, 3], num_classes=n_cls).to(device)
    total_params = sum(p.numel() for p in model.parameters())

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        momentum=args.momentum, 
        weight_decay=args.wd
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=[100, 150], 
        gamma=0.1
    )

    # Start Logging
    logger.info(f"Weights Paths        : {ckpt_name} | {best_name}")
    logger.info(f"Dataset              : {args.dataset.upper()}")
    logger.info(f"Dataset Loading Time : {data_load_time:.2f}s")
    logger.info(f"Total Parameters     : {total_params / 1e6:.2f}M")
    logger.info(f"Optimizer LR         : {args.lr}")
    logger.info(f"Momentum             : {args.momentum}")
    logger.info(f"Weight Decay         : {args.wd}")
    logger.info("-"*85)
    logger.info(f"{'Epoch':<6} {'Loss':<12} {'Train Acc(%)':<15} {'Acc Top 5 (%)':<15} {'Acc Top 1 (%)':<15} {'Time(s)':<10}")
    logger.info("-"*85)


    best_acc = 0
    for epoch in tqdm(range(args.epochs)):
        model.train()
        start_time = time.time()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        model.eval()
        top1_acc, top5_acc = 0, 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                acc1, acc5 = get_accuracy(outputs, targets, topk=(1, 5))
                top1_acc += acc1[0].item()
                top5_acc += acc5[0].item()

        top1_acc /= len(testloader)
        top5_acc /= len(testloader)
        epoch_time = time.time() - start_time
        train_acc = 100. * correct / total

        logger.info(f"{epoch+1:<6} {running_loss/len(trainloader):<12.4f} {train_acc:<15.2f} {top5_acc:<15.2f} {top1_acc:<15.2f} {epoch_time:<10.2f}")

        # Save Weights
        is_best = top1_acc > best_acc
        best_acc = max(top1_acc, best_acc)
        state = {
            'model': model.state_dict(),
            'acc': top1_acc,
            'epoch': epoch,
            'dataset': args.dataset
        }

        torch.save(state, ckpt_name)
        if is_best:
            torch.save(state, best_name)

        scheduler.step()

if __name__ == '__main__':
    main()
