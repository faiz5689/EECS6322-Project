import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import mobilenet_v2
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import argparse

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = './data'

# CIFAR-10 dataset
def get_dataset(cifar10=True, cifar100=False):
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

    if cifar10:
        train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                                    download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                                    download=False, transform=transform_test)
    elif cifar100:
        train_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True,
                                                    download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False,
                                                    download=False, transform=transform_test)
    else:
        raise ValueError('Please specify a dataset')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128,
                                            shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128,
                                            shuffle=False, num_workers=4)

    return train_loader, test_loader

class SimAM(nn.Module):
    def __init__(self, coeff_lambda=1e-4):
        super(SimAM, self).__init__()
        self.coeff_lambda = coeff_lambda

    def forward(self, X):
        """
        X: input tensor with shape (batch_size, num_channels, height, width)
        """
        assert X.dim() == 4, "shape of X must have 4 dimension"

        # spatial size
        n = X.shape[2] * X.shape[3] - 1
        n = 1 if n==0 else n

        # square of (t - u)
        d = (X - X.mean(dim=[2,3], keepdim=True)).pow(2)

        # d.sum() / n is channel variance
        v = d.sum(dim=[2,3], keepdim=True) / n

        # E_inv groups all importance of X
        E_inv = d / (4 * (v + self.coeff_lambda)) + 0.5

        # return attended features
        return X * F.sigmoid(E_inv)

# Define the model
def get_mobilenetv2(simam=False):
    class Block(nn.Module):
        '''expand + depthwise + pointwise'''
        def __init__(self, in_planes, out_planes, expansion, stride, attention):
            super(Block, self).__init__()
            self.stride = stride

            planes = expansion * in_planes
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
            if attention:
                self.attention = True
                self.simam = SimAM()
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn3 = nn.BatchNorm2d(out_planes)

            self.shortcut = nn.Sequential()
            if stride == 1 and in_planes != out_planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_planes),
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.conv2(out)
            if self.attention:
                out = self.simam(out)
            out = F.relu(self.bn2(out))
            out = self.bn3(self.conv3(out))
            out = out + self.shortcut(x) if self.stride==1 else out
            return out


    class MobileNetV2(nn.Module):
        # (expansion, out_planes, num_blocks, stride)
        cfg = [(1,  16, 1, 1),
            (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
            (6,  32, 3, 2),
            (6,  64, 4, 2),
            (6,  96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1)]

        def __init__(self, num_classes=10, attention=False):
            super(MobileNetV2, self).__init__()
            # NOTE: change conv1 stride 2 -> 1 for CIFAR10
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(32)
            self.layers = self._make_layers(in_planes=32, attention=attention)
            self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn2 = nn.BatchNorm2d(1280)
            self.linear = nn.Linear(1280, num_classes)

        def _make_layers(self, in_planes, attention):
            layers = []
            for expansion, out_planes, num_blocks, stride in self.cfg:
                strides = [stride] + [1]*(num_blocks-1)
                for stride in strides:
                    layers.append(Block(in_planes, out_planes, expansion, stride, attention))
                    in_planes = out_planes
            return nn.Sequential(*layers)

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layers(out)
            out = F.relu(self.bn2(self.conv2(out)))
            # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out

    model = MobileNetV2(attention=simam)
    return model.to(device)

def update_lr(iterations, optimizer, lr, lr_reduced32k, lr_reduced48k):
    if iterations >= 32000 and not lr_reduced32k:
        lr = lr / 10
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        lr_reduced32k = True
    elif iterations >= 48000 and not lr_reduced48k:
        lr = lr / 10
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        lr_reduced48k = True

    return lr, lr_reduced32k, lr_reduced48k

def log_tensorboard(writer, train_loss, train_accuracy, test_loss, test_accuracy, epoch, simam):
    if not simam:
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/Accuracy', train_accuracy, epoch)
        writer.add_scalar('Test/Loss', test_loss, epoch)
        writer.add_scalar('Test/Accuracy', test_accuracy, epoch)
    else:
        writer.add_scalar('TrainSimAM/Loss', train_loss, epoch)
        writer.add_scalar('TrainSimAM/Accuracy', train_accuracy, epoch)
        writer.add_scalar('TestSimAM/Loss', test_loss, epoch)
        writer.add_scalar('TestSimAM/Accuracy', test_accuracy, epoch)

# Loss and optimizer
def train(model,
          train_loader,
          test_loader,
          device,
          simam,
          writer=None,
          learning_rate=0.1):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    # Train the model
    total_steps = len(train_loader)
    num_epochs = 300
    iterations = 0
    lr_reduced32k = False
    lr_reduced48k = False

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)

            # Backward and optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iterations += 1
            learning_rate, lr_reduced32k, lr_reduced48k = update_lr(iterations,
                                                                    optimizer,
                                                                    learning_rate,
                                                                    lr_reduced32k,
                                                                    lr_reduced48k)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss/total_steps
        train_accuracy = 100.*correct/total

        test_accuracy, test_loss = test(model, test_loader, device)

        if writer is not None:
            log_tensorboard(writer, train_loss, train_accuracy, test_loss, test_accuracy, epoch, simam)


        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.2f}%, Test Loss: {:.4f}, Test Accuracy: {:.2f}%'.format(
            epoch+1, num_epochs, train_loss, train_accuracy, test_loss, test_accuracy))

def test(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        test_loss = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            test_loss += F.cross_entropy(outputs, labels).item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        average_loss = test_loss / len(test_loader)

        return accuracy, average_loss


def main(args):
    writer = SummaryWriter(log_dir='logs')
    train_loader, test_loader = get_dataset(cifar10=args.cifar10, cifar100=args.cifar100)

    model = get_mobilenetv2(args.simam)
    train(model, train_loader, test_loader, device, args.simam, writer=writer)

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MobileNetV2')
    parser.add_argument('--cifar10', action='store_true', help='Use CIFAR-10 dataset')
    parser.add_argument('--cifar100', action='store_true', help='Use CIFAR-100 dataset')
    parser.add_argument('--simam', action='store_true', help='Use SIMAM')
    args = parser.parse_args()
    
    main(args)