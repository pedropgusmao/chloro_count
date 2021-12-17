import torch
import itertools
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import utils
from datasets import ChloroplastsDataset
from pathlib import Path
from train import get_instance_segmentation_model, get_transform
import pickle
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms
from typing import Dict, List, Optional
from sklearn.cluster import AgglomerativeClustering

from matplotlib import pyplot as plt


def get_unique_labels(mask_image):
    labels = np.unique(mask_image)


def plot_clustering(current_prediction_mask, X_red, predicted_labels, idx, title=None):

    # plt.figure(figsize=(6, 6))
    plt.imshow(current_prediction_mask)
    for i in range(X_red.shape[0]):
        plt.text(
            X_red[i, 1],  # / 512,
            X_red[i, 0],  # / 512,
            str(predicted_labels[i]),
            color=plt.cm.nipy_spectral(predicted_labels[i] / 20.0),
            fontdict={"weight": "bold", "size": 9},
        )

    # plt.xticks([])
    # plt.yticks([])
    # if title is not None:
    #    plt.title(title, size=17)
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.axis("off")
    plt.savefig(f"{title}_{idx}.png", bbox_inches="tight", pad_inches=0)
    plt.figure().clear()
    plt.close(),
    plt.cla()
    plt.clf()


def get_prediction_centers(np_prediction):
    flat_prediction = flatten_prediction(np_prediction)
    num_predictions, w, h = flat_prediction.shape
    list_centers = []
    for segmentation_idx in range(num_predictions):
        this_segmentation = flat_prediction[segmentation_idx]
        y, x = np.where(this_segmentation > 0)
        list_centers.append([np.mean(y), np.mean(x)])
    return list_centers


def predictions_to_numpy(prediction: List[Dict]):
    pass


def clean_min_score(np_prediction, np_scores, score_threshold=0.8):
    good_segmentations = np_scores >= score_threshold
    return np_prediction[good_segmentations]


def flatten_prediction(np_tensor):
    """Removes singleton dimension (channel=1) for predictions

    Args:
        np_tensor ([numpy.ndarray]): tensor of shape (num_predictions, 1, height, width)

    Returns:
        [numpy.ndarray]: tensor of shape (num_predictions, height, width)
    """
    squeezed_tensor = np_tensor.squeeze(axis=-3)
    c, w, h = squeezed_tensor.shape
    return squeezed_tensor


def prediction_to_mask(np_tensor, min_img_value=0.1):
    """Transforms a single prediction tensor in list of individual masks,
    one for each prediction.

    Args:
        np_tensor (numpy.ndarray): Prediction tensor of the form (num_predictions, height, width)
        min_img_value (float): Threshold defining the image likelihood. This is different from the score.

    Returns:
        [type]: [description]
    """
    flat_prediction = flatten_prediction(np_tensor)
    num_predictions, w, h = flat_prediction.shape
    mask = np.zeros(shape=(w, h), dtype=np.uint8)
    for segmentation_idx in range(num_predictions):
        this_segmentation = flat_prediction[segmentation_idx]
        non_zero_indices = np.where(this_segmentation >= min_img_value)
        mask[non_zero_indices] = 255
    return mask


def read_img(image_path: Path, device) -> torch.Tensor:
    """Reads path of a single image and converts it into a Tensor

    Args:
        image_path (Path): Path to image.
        device (str): Which device to use.

    Returns:
        torch.Tensor: Output tensor values [0-1] [C,H,W]
    """
    pil_image = Image.open(image_path)
    Loader = transforms.Compose([transforms.ToTensor()])
    return Loader(pil_image).to(device)


def read_img_planes(list_paths: List[Path], device):
    """Applies read_img to a list of paths

    Args:
        list_paths (List[Path]): List of paths to images.
        device (str): Which device to use.

    Returns:
        [Lits[torch.Tensor]]: List of output tensors.
    """
    all_planes = [read_img(x, device) for x in list_paths]
    return all_planes


def evaluate(model, dataloader, device):
    for image_idx, (images, targets) in enumerate(dataloader):
        prediction = model(images)
        pred_masks = prediction[0]["masks"]
        pred_boxes = prediction[0]["boxes"]
        target_boxes = targets[0]["boxes"]


def main():
    # Choose device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load model
    num_classes = 2
    model = get_instance_segmentation_model(num_classes)
    model.load_state_dict(torch.load("saved_models/chloro_count_chloroplasts.pth"))
    # model.load_state_dict(torch.load("saved_models/chloro_count.pth"))
    model.eval()
    model.to(device)

    # Get list of images for a given cell.
    cell_name = "22A41#7 - Series003"
    this_cell = [
        f"/home/pedro/repos/chloro_count/data/images/{cell_name}_{x}.tif"
        for x in range(11)
    ]
    print(f"Cell {cell_name} contains the following planes:")

    # Generate masks for each plane in the cell:
    all_planes_tensors = read_img_planes(this_cell, device)
    with torch.no_grad():
        predictions = model(
            all_planes_tensors
        )  # returns a list of predictions one for each plane in a cell
    # a single element of predictions (ex. predictions[0])
    # print(predictions[0]["scores"])

    ## Now with the list of predictions, one for each plane,
    ## turn them into list. Keep the scores
    np_predictions = [
        pred["masks"].cpu().detach().numpy()
        for pred in predictions  # prediction as np.ndarray
    ]
    np_predictions_scores = [
        pred["scores"].cpu().detach().numpy()
        for pred in predictions  # prediction as np.ndarray
    ]
    ## Filter predictions based on scores
    np_predictions = [
        clean_min_score(pred, score, score_threshold=0.8)
        for (pred, score) in zip(np_predictions, np_predictions_scores)
    ]
    ## Get centers now that we have filtered
    predictions_centers = [get_prediction_centers(pred) for pred in np_predictions]

    ## Get single numpy array with all centers
    np_predictions_centers = np.array(
        list(itertools.chain.from_iterable(predictions_centers))
    )
    resp = prediction_to_mask(np_predictions[0])
    img = Image.fromarray(resp, "L")
    img.save("lala.png")

    ## Look for clustering based on distance matrix.
    real_labels = np.zeros((np_predictions_centers.shape[0]))

    ## Look for clustering based on distance matrix.
    # LOOK AT https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
    # for `fit`, affinity `precomputed` and pass a distance matrix with Infinity where centers are in the same plane
    # calculate within a plane the minimum distance between all segmentation to define `distance_threshold`.
    # Maybe do clustering sequentially using pairs of consecutive planes.

    # Fit
    for linkage in ["ward"]:  # , "average", "complete", "single"]:
        clustering = AgglomerativeClustering(
            n_clusters=None, affinity="euclidean", distance_threshold=10
        )
        clustering.fit(np_predictions_centers)
        len_planes = [len(plane) for plane in predictions_centers]
        len_planes.insert(0, 0)
        len_planes = np.cumsum(len_planes)
        for idx, plane in enumerate(predictions_centers):
            start = len_planes[idx]
            end = len_planes[idx + 1]
            labels = clustering.labels_[start:end]
            plot_clustering(
                current_prediction_mask=prediction_to_mask(np_predictions[idx]),
                X_red=np.array(plane),
                predicted_labels=labels,
                idx=idx,
                title=linkage,
            )


if __name__ == "__main__":
    main()
