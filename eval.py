import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import utils
from datasets import ChloroplastsDataset
from pathlib import Path
from train import get_model_instance_segmentation, get_transform
import pickle


def evaluate(model, dataloader, device):
    for image_idx, (images, targets) in enumerate(dataloader):
        prediction = model(images)
        pred_masks = prediction[0]["masks"]
        pred_boxes = prediction[0]["boxes"]
        target_boxes = targets[0]["boxes"]


def main():
    # Choose device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # split the dataset in train and test set
    dataset_cell = ChloroplastsDataset(
        dataset_dir=Path("/home/pedro/repos/chloro_count/data/"),
        dataset_file=Path("/home/pedro/repos/chloro_count/datasets/")
        / "cells"
        / "0.txt",
    )

    # define training and validation data loaders
    dataloader = torch.utils.data.DataLoader(
        dataset_cell,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )

    # Load model
    num_classes = 2
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load("saved_models/chloro_count.pth"))
    model.eval()
    model.to(device)

    # Generate all masks:
    for idx, (images, targets) in enumerate(dataloader):
        images, targets = images.to(device), targets.to(device)
        prediction = model(images)
        pred_masks = prediction[0]["masks"]
        pred_boxes = prediction[0]["boxes"]
        target_boxes = targets[0]["boxes"]

        remaining_target_idx = [x for x in range(target_boxes.shape[0])]
        # Check threshold:
        for pred_mask_idx, this_pred_mask in enumerate(pred_masks):
            """For each predicted mask, find the pixel coordinates of that mask"""


if __name__ == "__main__":
    main()
