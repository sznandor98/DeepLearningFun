import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import time
import matplotlib.pyplot as plt
import numpy as np

from common import get_device

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, writer, is_aux_output, which_epoch, device):
    """
    Trains one epoch on a given model with given train loader, criterion, optimizer and scheduler.
    """
    # Set device to training mode.
    model.train(True)
    running_loss = 0.0
    batch_loss = 0.0
    running_corrects = 0
    # Measure time between batches
    t = time.process_time()
    for i, data in enumerate(train_loader):
        # Move inputs to device.
        inputs, labels = data[0].to(device), data[1].to(device)
        # Zero the parameter gradients
        optimizer.zero_grad()

        # For inception v3 net
        if is_aux_output:
            # Get output
            outputs, aux_outputs = model(inputs)
            # Find output maximum which is the result class
            _, predictions = torch.max(outputs, 1)
            # Calculate the loss of output
            loss1 = criterion(outputs, labels)
            # Loss of aux output
            loss2 = criterion(aux_outputs, labels)
            # Construct a final loss from the two losses
            loss = loss1 + 0.4*loss2
        else:
            # forward + backward + optimize
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        # Backprop with loss
        loss.backward()
        # Step one
        optimizer.step()
        # statistics
        running_loss += loss.item()
        batch_loss += loss.item()
        running_corrects += (predictions == labels).sum().item()
        if i % 100 == 99:
            print(
                f"[{which_epoch+1:d}, {i+1:5d}] Loss: {batch_loss/100:.3f}, time: {time.process_time()-t:.3f} sec")
            t = time.process_time()
            # ...log the running loss
            writer.add_scalar("training_loss",
                              batch_loss / 100,
                              which_epoch * len(train_loader) + i)
            batch_loss = 0.0

    # Step one with scheduler after one epoch
    scheduler.step()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100*(running_corrects / len(train_loader.dataset))
    print(f"Loss: {epoch_loss:.4f} Acc: {epoch_acc:.1f} %")

    return model


def validate_model(model, valid_loader, criterion, is_aux_output, device):
    """
    Calculates validation data
    """
    # Model to evaluation mode
    model.eval()
    valid_loss = 0.0
    corrects = 0
    for inputs, labels in valid_loader:
        # Inputs to device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Get output
        outputs = model(inputs)
        # Get result according to maximum value
        _, predictions = torch.max(outputs, 1)
        # Calculate loss from output and input labels
        loss = criterion(outputs, labels)
        valid_loss += loss.item()
        # Count correct predictions
        corrects += (predictions == labels).sum().item()
    valid_acc = corrects / len(valid_loader.dataset)
    valid_loss = valid_loss / len(valid_loader)
    print(f"Validation loss: {valid_loss:.3f}")
    print(f"Validation acc: {100 * valid_acc:.1f} %")
    return valid_loss, valid_acc


def train_model(model, train_loader, valid_loader, num_epochs, criterion, optimizer, scheduler, writer, is_aux_output=False):
    """
    Trains model with a given train loader, epoch number, criterion function, optimizer and scheduler.
    """
    device = get_device()
    # Move model to device
    model.to(device)
    since = time.process_time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 50)

        # Do one epoch of training
        model = train_one_epoch(model, train_loader, criterion,
                                optimizer, scheduler, writer, is_aux_output, epoch, device)

        # Calculate validation stats
        valid_loss, valid_acc = validate_model(
            model, valid_loader, criterion, is_aux_output, device)

        writer.add_scalar("validation_loss", valid_loss, epoch)
        writer.add_scalar("validation_acc", valid_acc, epoch)

        # Save best model
        if valid_acc > best_acc:
            best_acc = valid_acc
            #best_model_wts = model.state_dict().copy()

    time_elapsed = time.process_time() - since
    print(
        f"Training complete in {time_elapsed//60:.0f} min {time_elapsed%60:.0f} sec")
    print(f"Best val Acc: {100 * best_acc:.1f} %")


def add_pr_curve_tensorboard(class_index, test_probs, test_label, classes, writer, global_step=0):
    tensorboard_truth = test_label == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(classes[class_index],
                        tensorboard_truth,
                        tensorboard_probs,
                        global_step=global_step)
    writer.close()


def test_model(model, data_loader, classes, writer, is_aux_output=False):
    device = get_device()
    model.to(device)
    model.eval()
    correct = 0
    total = 0

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    class_probs = []
    class_label = []

    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data[0].to(device), data[1].to(device)

            if is_aux_output:
                outputs, _ = model(inputs)
            else:
                outputs = model(inputs)

            class_probs_batch = [F.softmax(el, dim=0) for el in outputs]
            class_probs.append(class_probs_batch)
            class_label.append(labels)

            _, predictions = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_label = torch.cat(class_label)
    for i in range(len(classes)):
        add_pr_curve_tensorboard(i, test_probs, test_label, classes, writer)

    print(
        f"Accuracy of the network on the 10000 test images: {100*correct/total:.1f} %")

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f"Accuracy for class {classname:5s} is: {accuracy:.1f} %")


class SimpleCNN(nn.Module):
    """
    Simple CNN for 32x32 images as input and 10 output (CIFAR)
    conv1->ReLu->pool->
    conv2->ReLu->pool->
    flatten->
    FC1->ReLu->
    FC2->ReLu->
    FC3->output
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
