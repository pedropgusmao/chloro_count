from turtle import shape
import numpy as np
import torch
import itertools
from pathlib import Path
from readlif.reader import LifFile
from train import get_instance_segmentation_model, get_transform
import os, datetime
from PIL import Image
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt


def plot_clustering(original_image, masks, centers, labels, save_path, alpha=0.8):

    np_image = alpha * original_image.copy()
    for idx, this_mask in enumerate(masks):
        np_image += (1.0 - alpha) * this_mask

    np_image_rgb = np_image.transpose((1, 2, 0))
    plt.figure()
    plt.imshow(np_image_rgb)
    for idx, center in enumerate(centers):
        plt.text(
            center[1],  # /512,
            center[0],  # / 512,
            str(labels[idx]),
            # color=plt.cm.nipy_spectral(predicted_labels[i] / 20.0),
            # fontdict={"weight": "bold", "size": 9},
            fontdict={"size": 9},
        )

    # plt.xticks([])
    # plt.yticks([])
    # if title is not None:
    plt.title(f"Num. Chloroplasts: {len(np.unique(labels))}", size=15)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # img = Image.fromarray((255 * np_image_rgb).astype(np.uint8), "RGB")
    # b, g, r = img.split()
    # img = Image.merge("RGB", (r, g, b))
    # img.save(save_path)
    plt.savefig(save_path)


def filter_and_binarize(predictions, scores, threshold):
    segmented_image = np.zeros_like(predictions[0][0], dtype=np.uint8)
    for idx, score in enumerate(scores):
        if score >= threshold:
            segmented_image[predictions[idx][0] > 0.5] = idx + 1

    return segmented_image


def get_prediction_centers(np_prediction):
    flat_prediction = np_prediction.squeeze(axis=-3)
    num_predictions, w, h = flat_prediction.shape
    list_centers = []
    for segmentation_idx in range(num_predictions):
        this_segmentation = flat_prediction[segmentation_idx]
        y, x = np.where(this_segmentation > 0)
        list_centers.append([np.mean(y), np.mean(x)])
    return list_centers


def save_images(list_images, prefix, save_dir):
    for idx, np_img in enumerate(list_images):
        im = Image.fromarray(np_img)
        im.save(f"{save_dir}/{prefix}_{idx}.png")


def get_cell_np(cell_lif_image):

    scale = cell_lif_image.scale
    pixel_width = 1.0 / scale[0]  # um/px
    pixel_height = 1.0 / scale[1]
    pixel_depth = 0.988
    voxel_density = pixel_depth * pixel_width * pixel_height

    tmp = []
    for plane_idx in range(cell_lif_image.nz):
        tmp.append(
            [
                np.array(channel_image, dtype=np.float32)
                for channel_image in cell_lif_image.get_iter_c(t=0, z=int(plane_idx))
            ]
        )
    np_cell = np.array(tmp, dtype=np.float32) / 255.0

    return np_cell


def get_segments(path_to_model, cell_img_stack, threshold=0.5):
    # Load models
    model = get_instance_segmentation_model(num_classes=2)
    model.load_state_dict(torch.load(path_to_model))
    model.eval()

    torch_tensor = torch.Tensor(cell_img_stack)
    outputs = model(torch_tensor)  # (N,*,H,W)
    segments = []  # N
    for output in outputs:
        scores = output["scores"].detach().numpy()  # (*)
        masks = output["masks"].detach().numpy()  # (*,H,W)
        masks = masks[scores > threshold, :, :]
        # masks[masks > 0] = 1.0
        segments.append(masks)

    return segments


def main():
    # choose a lif file and a cell id
    path_lif_file = "/home/pedro/repos/chloro_count/data/lif/22A41#7.lif"
    cell_id = 0

    # Create save paths
    exp_path = Path("experiments") / datetime.datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    # Load file
    try:
        lif_file = LifFile(path_lif_file)
    except Exception as e:
        print(e)

    # Process Cell
    # print(f"Num cells: {lif_file.image_list}")
    lif_image = lif_file.get_image(cell_id)
    lif_image_path = lif_image.path.replace("#", "_")
    lif_image_name = lif_image.name
    exp_path_imgs = exp_path / lif_image_path / lif_image_name
    exp_path_imgs.mkdir(parents=True, exist_ok=True)

    cell_img_stack = get_cell_np(lif_image)  # (N,C,H,W)

    segmented_chloroplasts = (
        get_segments(  # list of N sets of predictions(variable length)
            cell_img_stack=cell_img_stack,
            path_to_model="saved_models/chloro_count_chloroplasts.pth",
        )
    )
    segmented_sheath = get_segments(
        cell_img_stack=cell_img_stack,
        path_to_model="saved_models/chloro_count_bundle_sheath.pth",
    )

    # Select chloroplasts that are inside the cell
    # TODO

    # Get centers from that we have filtered
    predictions_centers = [
        get_prediction_centers(plane) for plane in segmented_chloroplasts
    ]

    # Get single numpy array with all centers and plane index
    # np_predictions_centers = np.array(
    #    list(itertools.chain.from_iterable(predictions_centers))
    # )
    # print(np_predictions_centers.shape)
    predictions_center_with_idx = []
    for idx, plane in enumerate(predictions_centers):
        for center in plane:
            center_idx = [center[0], center[1], idx]
            predictions_center_with_idx.append(center_idx)
    np_predictions_centers_with_idx = np.array(predictions_center_with_idx)

    num_centers = np_predictions_centers_with_idx.shape[0]
    DistMat = np.full((num_centers, num_centers), 100000, dtype=np.float32)
    for i in range(num_centers - 1):
        center_i = np_predictions_centers_with_idx[i][:2]
        plane_i = np_predictions_centers_with_idx[i][2]
        for j in range(i, num_centers):
            center_j = np_predictions_centers_with_idx[j][:2]
            plane_j = np_predictions_centers_with_idx[j][2]
            if plane_i != plane_j:
                dist = np.linalg.norm(center_i - center_j)
                DistMat[i, j] = dist
                DistMat[j, i] = dist

    ## Look for clustering based on distance matrix.
    # LOOK AT https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
    # for `fit`, affinity `precomputed` and pass a distance matrix with Infinity where centers are in the same plane
    # calculate within a plane the minimum distance between all segmentation to define `distance_threshold`.
    # Maybe do clustering sequentially using pairs of consecutive planes.

    # for linkage in ["ward"]:  # , "average", "complete", "single"]:
    clustering = AgglomerativeClustering(
        n_clusters=None,
        linkage="complete",
        affinity="precomputed",
        distance_threshold=10,
    )
    clustering.fit(DistMat)
    print(f"Found {clustering.n_clusters_} chloroplasts.")

    for idx, plane in enumerate(segmented_chloroplasts):
        cell_image = Image.fromarray(
            np.transpose(255 * cell_img_stack[idx], (1, 2, 0)).astype(np.uint8)
        )
        b, r, g = cell_image.split()
        cell_image = Image.merge("RGB", (r, g, b))
        these_indices = np_predictions_centers_with_idx[:, 2] == idx
        these_labels = clustering.labels_[these_indices]
        these_centers = np_predictions_centers_with_idx[these_indices, :2]
        plot_clustering(
            original_image=cell_img_stack[idx],
            masks=segmented_chloroplasts[idx],
            centers=these_centers,
            labels=these_labels,
            save_path=exp_path_imgs / f"{idx:02d}.png",
            alpha=0.8,
        )
    # For each cluster label, calculate volume


if __name__ == "__main__":
    main()
