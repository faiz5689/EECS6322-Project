import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import mobilenet_v2
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() elif 'mps' if torch.backends.mps.isavailable() else 'cpu')
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# CIFAR-10 dataset
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

train_dataset = torchvision.datasets.CIFAR10(root='./Project/data', train=True,
                                             download=False, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128,
                                           shuffle=True, num_workers=4)

test_dataset = torchvision.datasets.CIFAR10(root='./Project/data', train=False,
                                            download=False, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128,
                                          shuffle=False, num_workers=4)

# Define the model
model = mobilenet_v2(pretrained=True)
model.classifier[-1] = nn.Linear(in_features=1280,
                                 out_features=10)
model = model.to(device)


# Loss and optimizer
learning_rate = 0.1
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

# Train the model
total_steps = len(train_loader)
num_epochs = 300
iterations = 0
lr_reduced32k = False
lr_reduced48k = False
lr_reduced64k = False


if __name__ == "__main__":
    print(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for param_group in optimizer.param_groups:
            print(f"epoch: {epoch}, lr: {param_group['lr']}")
            break
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    #        iterations += 1
            if iterations >= 32000 and not lr_reduced32k:
                learning_rate = learning_rate / 10
                print(f'new learning rate: {learning_rate}')
                for param_group in optimizer.param_groups:
                    param_group["lr"] = learning_rate
                lr_reduced32k = True
            elif iterations >= 48000 and not lr_reduced48k:
                learning_rate = learning_rate / 10
                print(f'new learning rate: {learning_rate}')
                for param_group in optimizer.param_groups:
                    param_group["lr"] = learning_rate
                lr_reduced48k = True
    #        elif iterations >= 64000 and not lr_reduced64k:
    #            learning_rate = learning_rate / 10
    #            print(f'new learning rate: {learning_rate}')
    #            for param_group in optimizer.param_groups:
    #                param_group["lr"] = learning_rate
    #            lr_reduced64k = True

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        print('Epoch [{}/{}], Loss: {:.4f}, Train Accuracy: {:.2f}%'.format(
            epoch+1, num_epochs, running_loss/total_steps, 100.*correct/total))

    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

