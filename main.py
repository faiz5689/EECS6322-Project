import torch
from torch.utils.tensorboard import SummaryWriter
import argparse

from dataset import get_dataset
from models import get_model

from train_test import train

def main(args):
    logdir = 'logs/' + args.arch + ('_simam' if args.simam else '') + ('_' + args.dataset)
    writer = SummaryWriter(log_dir=logdir)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader, test_loader = get_dataset(dataset=args.dataset)

    model = get_model ( arch=args.arch, 
                        num_classes=(10 if args.dataset=='cifar10' else 100), 
                        depth=args.depth,
                        widen_factor=args.widen_factor,
                        simam=args.simam )
    
    train(model, train_loader, test_loader, device, args.simam, writer=writer)

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model on CIFAR10 or CIFAR100')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset')
    parser.add_argument('--arch', type=str, default='mobilenetv2', help='Architecture')
    parser.add_argument('--depth', type=int, default=20, help='Depth of the network')
    parser.add_argument('--widen_factor', type=int, default=10, help='Wide factor for WideResNet')
    parser.add_argument('--simam', action='store_true', help='Use SIMAM')
    args = parser.parse_args()
    
    main(args)
