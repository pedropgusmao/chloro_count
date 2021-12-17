import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import utils
import segment_trans as T
from datasets import ChloroplastsDataset
from pathlib import Path

from engine import train_one_epoch, evaluate
import utils
import segment_trans as T


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    # transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations

    # split the dataset in train and validation set
    trainset = ChloroplastsDataset(
        data_dir=Path("/home/pedro/repos/chloro_count/data/"),
        dataset_file=Path("/home/pedro/repos/chloro_count/data/") / "train_images.txt",
        transforms=get_transform(train=True),
    )
    validation = ChloroplastsDataset(
        data_dir=Path("/home/pedro/repos/chloro_count/data/"),
        dataset_file=Path("/home/pedro/repos/chloro_count/data/") / "val_images.txt",
        transforms=get_transform(train=False),
    )

    # define training and validation data loaders
    dataloader_train = torch.utils.data.DataLoader(
        trainset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )

    dataloader_val = torch.utils.data.DataLoader(
        validation,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )

    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=0.005, momentum=0.9, weight_decay=0.0005
    )  # original lr=0.005
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(
            model, optimizer, dataloader_train, device, epoch, print_freq=327
        )

        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, dataloader_val, device=device)

    # Save model
    torch.save(model.state_dict(), "./saved_models/chloro_count_bundle_sheath.pth")


if __name__ == "__main__":
    main()
