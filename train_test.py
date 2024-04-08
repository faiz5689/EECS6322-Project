import torch
import torch.optim as optim
import torch.nn.functional as F

from utils import update_lr, log_tensorboard

def train ( model,
            train_loader,
            test_loader,
            device,
            simam,
            writer=None,
            learning_rate=0.1 ):
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