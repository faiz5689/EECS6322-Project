import torch
import torchvision
import torchvision.transforms as transforms

data_dir = './data'

def get_dataset(dataset='cifar10'):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                                    download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                                    download=False, transform=transform_test)
    elif dataset == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True,
                                                    download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False,
                                                    download=False, transform=transform_test)
    else:
        raise ValueError('Invalid dataset name. Must be either cifar10 or cifar100.')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128,
                                            shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128,
                                            shuffle=False, num_workers=4)

    return train_loader, test_loader