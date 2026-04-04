import torch
import torchvision
import torchvision.transforms as transforms

# Data loaders
def get_dataloaders(dataset: str, data_dir: str, batch_size: int,
                    num_workers: int = 4):
    
    # Normalisation statistics
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    DatasetClass = (torchvision.datasets.CIFAR10
                    if dataset == 'cifar10'
                    else torchvision.datasets.CIFAR100)
    
    num_classes = 10 if dataset == 'cifar10' else 100

    train_set = DatasetClass(root=data_dir, train=True,
                              download=True, transform=train_transform)
    
    val_set   = DatasetClass(root=data_dir, train=False,
                              download=True, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    
    val_loader   = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, num_classes
