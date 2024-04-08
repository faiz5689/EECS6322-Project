import math
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
def get_preresnet(simam=False):
    def conv3x3(in_planes, out_planes, stride=1):
        "3x3 convolution with padding"
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=1, bias=False)


    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, inplanes, planes, stride=1, downsample=None, attention=False):
            super(BasicBlock, self).__init__()
            self.bn1 = nn.BatchNorm2d(inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv2 = conv3x3(planes, planes)
            if attention:
                self.attention = True
                self.simam = SimAM()
            else:
                self.attention = False
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            residual = x

            out = self.bn1(x)
            out = self.relu(out)
            out = self.conv1(out)
            out = self.bn2(out)
            out = self.relu(out)
            out = self.conv2(out)
            if self.attention:
                out = self.simam(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual

            return out


    class Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, inplanes, planes, stride=1, downsample=None, attention=False):
            super(Bottleneck, self).__init__()
            self.bn1 = nn.BatchNorm2d(inplanes)
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
            if attention:
                self.attention = True
                self.simam = SimAM()
            else:
                self.attention = False
            self.bn3 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            residual = x

            out = self.bn1(x)
            out = self.relu(out)
            out = self.conv1(out)

            out = self.bn2(out)
            out = self.relu(out)
            out = self.conv2(out)
            if self.attention:
                out = self.simam(out)

            out = self.bn3(out)
            out = self.relu(out)
            out = self.conv3(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual

            return out


    class PreResNet(nn.Module):

        def __init__(self, depth, num_classes=1000, block_name='BasicBlock', attention=False):
            super(PreResNet, self).__init__()
            # Model type specifies number of layers for CIFAR-10 model
            if block_name.lower() == 'basicblock':
                assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
                n = (depth - 2) // 6
                block = BasicBlock
            elif block_name.lower() == 'bottleneck':
                assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
                n = (depth - 2) // 9
                block = Bottleneck
            else:
                raise ValueError('block_name shoule be Basicblock or Bottleneck')

            self.inplanes = 16
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                                bias=False)
            self.layer1 = self._make_layer(block, 16, n, attention=attention)
            self.layer2 = self._make_layer(block, 32, n, stride=2, attention=attention)
            self.layer3 = self._make_layer(block, 64, n, stride=2, attention=attention)
            self.bn = nn.BatchNorm2d(64 * block.expansion)
            self.relu = nn.ReLU(inplace=True)
            self.avgpool = nn.AvgPool2d(8)
            self.fc = nn.Linear(64 * block.expansion, num_classes)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

        def _make_layer(self, block, planes, blocks, stride=1, attention=False):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=stride, bias=False),
                )

            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample, attention=attention))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, attention=attention))

            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)

            x = self.layer1(x)  # 32x32
            x = self.layer2(x)  # 16x16
            x = self.layer3(x)  # 8x8
            x = self.bn(x)
            x = self.relu(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return x


    def preresnet(**kwargs):
        """
        Constructs a ResNet model.
        """
        return PreResNet(**kwargs)

    model = preresnet(depth=20, num_classes=100, attention=simam)
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
    if args.cifar10:
        log_dir = 'logs_cifar10_preresnet'
    else:
        log_dir = 'logs_cifar100_preresnet'
    writer = SummaryWriter(log_dir=log_dir)
    train_loader, test_loader = get_dataset(cifar10=args.cifar10, cifar100=args.cifar100)

    model = get_preresnet(args.simam)
    train(model, train_loader, test_loader, device, args.simam, writer=writer)

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PreResNet')
    parser.add_argument('--cifar10', action='store_true', help='Use CIFAR-10 dataset')
    parser.add_argument('--cifar100', action='store_true', help='Use CIFAR-100 dataset')
    parser.add_argument('--simam', action='store_true', help='Use SIMAM')
    args = parser.parse_args()

    main(args)
