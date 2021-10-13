import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from common import get_CIFAR10, get_PennFudan
from classifiers import SimpleCNN, train_model, test_model
from detection import MaskRCNN, train_mrcnn, test_mrcnn, eval_mrcnn

def main1():
    trainloader, validloader, testloader, classes = get_CIFAR10(32)

    net = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    mynet_writer = SummaryWriter('runs/SimpleCNN')

    test_model(net, testloader, classes, mynet_writer)

    train_model(net, trainloader, validloader, 50, criterion,
                optimizer, scheduler, mynet_writer)


def main():
    trainloader, testloader, classes = get_PennFudan(2, "PennFudanPed")

    mrcnn = MaskRCNN(len(classes))
    mrcnn_params = [p for p in mrcnn.parameters() if p.requires_grad]
    mrcnn_o =optim.SGD(mrcnn_params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    mrcnn_s = optim.lr_scheduler.StepLR(mrcnn_o, step_size=3, gamma=0.1)

    eval_mrcnn(mrcnn, classes, "PennFudanPed/PNGImages/FudanPed00044.png", "/content/testimg.png")

    coco_eval = test_mrcnn(mrcnn, testloader)

    train_mrcnn(mrcnn, trainloader, 10, mrcnn_o, mrcnn_s)

if __name__ == "__main__":
    main()
