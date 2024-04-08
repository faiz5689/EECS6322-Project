
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
    writer.add_scalar(('Train'+ ('SimAM' if simam else '') +'/Loss'), train_loss, epoch)
    writer.add_scalar(('Train'+ ('SimAM' if simam else '') +'/Accuracy'), train_accuracy, epoch)
    writer.add_scalar(('Test'+ ('SimAM' if simam else '') +'/Loss'), test_loss, epoch)
    writer.add_scalar(('Test'+ ('SimAM' if simam else '') +'/Accuracy'), test_accuracy, epoch)
