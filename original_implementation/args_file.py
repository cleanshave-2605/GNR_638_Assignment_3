import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training for iResNets')
parser.add_argument('--data', metavar='DIR', default='./data',
                    help='path to dataset (downloaded automatically if not present)')
parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'],
                    help='dataset to use: cifar10 (default) or cifar100')
parser.add_argument('-a', '--arch', metavar='ARCH', default='iresnet',
                    help='model architecture (default: iresnet)')
parser.add_argument('--result_path', default='results', type=str,
                    help='directory path where to save the results')
parser.add_argument('--model_depth', default=50, type=int,
                    help='depth of resnet (50 | 101 | 152)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run (default: 200 for CIFAR)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128 for CIFAR)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4 for CIFAR)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs.')
parser.add_argument('--n_classes', default=10, type=int,
                    help='Number of classes. (10 for CIFAR-10, 100 for CIFAR-100)')
parser.add_argument('--lr_scheduler', default='MultiStepLR', type=str,
                    help='The learning rate scheduler. Options: MultiStepLR')
parser.add_argument('--lr_steps', default=[100, 150], type=int, nargs="+",
                    metavar='LRSteps',
                    help='epochs to decay learning rate by 10 (default: [100, 150] for CIFAR)')
parser.add_argument('--lr_reduce_factor', default=0.1, type=float,
                    help='Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.')
parser.add_argument('--nesterov', action='store_true', default=False,
                    help='Nesterov momentum')
parser.add_argument('--zero_init_residual', action='store_true',
                    help='If true, zero-initialize the last BN in each residual branch.')
parser.add_argument('--groups', default=None, type=int,
                    help='the number of groups to split the spatial convolution')
parser.add_argument('--train_crop_size', default=32, type=int,
                    help='(Unused for CIFAR — kept for API compatibility)')
parser.add_argument('--val_resize', default=32, type=int,
                    help='(Unused for CIFAR — kept for API compatibility)')
parser.add_argument('--val_crop_size', default=32, type=int,
                    help='(Unused for CIFAR — kept for API compatibility)')
parser.add_argument('--save_freq', default=10, type=int,
                    help='Save a periodic checkpoint every N epochs. 0 to disable.')
