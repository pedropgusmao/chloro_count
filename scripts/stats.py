import torchvision
import torch
import numpy as np
import numpy.ma as ma
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw
from typing import List


def plot_img_with_ids(image, unique_ids, color_img):
    # Get a font
    fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMonoBold.ttf", 16)

    # convert image into RGB
    all_seg_255 = np.zeros_like(image)
    all_seg_255[image > 0] = 255
    base = Image.fromarray(all_seg_255)
    base = base.convert("RGBA")
    color_img = color_img.convert("RGBA")

    # make a blank image for the text, initialized to transparent text color
    blended = Image.blend(color_img, base, 0.5)
    txt = Image.new("RGBA", blended.size, (255, 255, 255, 0))

    # create mask:
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[np.where(image > 0)] = 200
    mask = Image.fromarray(mask).convert("L")

    # get a drawing context
    d = ImageDraw.Draw(txt)

    # Sort and remove background (0)
    unique_ids = np.sort(unique_ids)
    unique_ids = np.delete(unique_ids, np.where(unique_ids == 0))

    for k, idx in enumerate(unique_ids):
        center = get_center_from_idx(image, idx)

        # draw text, half opacity
        d.text(
            (center[1] - 4, center[0] - 6), str(k), font=fnt, fill=(0, 128, 255, 256)
        )

    out = Image.alpha_composite(blended, txt)
    # out = Image.composite(color_img, out, mask)
    # out = Image.blend(color_img, out, 0.5)

    return out


def get_idx_mapping(list_images: List[np.ndarray]):
    """Creates a list of unique segmentation ids from a list of images.

    Args:
        list_images (List[numpy.ndarray]): List of segmented images.

    Returns:
        numpy.ndarray: List of unique ids
    """
    individual_unique_ids = []
    for image in list_images:
        individual_unique_ids.append(np.unique(image))
    global_unique_ids = np.unique(np.concatenate(individual_unique_ids))

    return global_unique_ids


def get_center_from_idx(image: np.ndarray, idx: int):
    """Returns the coordinates (x, y) of a segmentation instance in the image.

    Args:
        image (numpy.ndarray): Grayscale image with 0-valued background and multiple
            segmentations, each of which with a given integer value.
        idx (int): Value of specific segmentation.

    Returns:
        numpy.ndarray: Center of selected segmentation.
    """
    _x, _y = np.where(image == idx)
    x_c = np.mean(_x, dtype=float)
    y_c = np.mean(_y, dtype=float)
    return np.array([x_c, y_c])


def get_instance_dist_between_two_image(current_img, next_img):
    # Get possible instance indices
    current_img_instances = np.unique(current_img)
    next_img_instances = np.unique(next_img)

    # Calculate distance with same index
    same_idx_distances = []
    common_instances = np.intersect1d(current_img_instances, next_img_instances)
    for idx in common_instances[1:]:  # Exclude 0 bg
        pass
        current_center = get_center_from_idx(current_img, idx)
        next_center = get_center_from_idx(next_img, idx)
        same_idx_distances.append(np.linalg.norm(current_center - next_center))

    # Best second index
    second_best_distances = []
    for idx in current_img_instances[1:]:  # Again, ignore bg
        best_dist = np.inf
        current_center = get_center_from_idx(current_img, idx)
        set_difference = np.setdiff1d(
            next_img_instances, np.array([idx])
        )  # remove only current value
        for candidate_idx in set_difference:
            next_center = get_center_from_idx(next_img, candidate_idx)
            current_dist = np.linalg.norm(current_center - next_center)
            if current_dist < best_dist:
                best_dist = current_dist
        second_best_distances.append(best_dist)

    return np.array(same_idx_distances), np.array(second_best_distances)


def get_instance_dist_for_cells_in_dataset(data_folder: Path, dataset_filename: Path):
    with open(dataset_filename, "r") as f:
        same_idx_dist, second_best_dist = [], []
        for cell_str in f:
            cell_str = cell_str.strip()
            list_imgs_in_cell = list(data_folder.glob(f"**/{cell_str}*.png"))
            list_images = [
                np.array(Image.open(data_folder / f"{cell_str}_{x}_chloroplast.png"))
                for x in range(len(list_imgs_in_cell))
            ]
            for img_idx in range(len(list_images) - 1):
                current_img = list_images[img_idx]
                next_img = list_images[img_idx + 1]
                _same_idx_dist, _second_best_dist = get_instance_dist_between_two_image(
                    current_img, next_img
                )
            same_idx_dist.append(_same_idx_dist)
            second_best_dist.append(_second_best_dist)
    return (np.concatenate(same_idx_dist), np.concatenate(second_best_dist))


if __name__ == "__main__":

    root_data = Path("/home/pedro/repos/chloro_count/data")
    chloro_masks = root_data / "masks" / "chloroplasts"
    dataset_filename = root_data / "train_images.txt"
    (
        same_idx_distances,
        second_closest_distances,
    ) = get_instance_dist_for_cells_in_dataset(chloro_masks, dataset_filename)

    bins = np.concatenate((np.linspace(0, 50, 100), np.linspace(51, 450, 20)))
    same_idx_hist, _ = np.histogram(same_idx_distances, bins=bins)
    same_idx_hist = same_idx_hist / np.sum(same_idx_hist, dtype=float)
    second_closest_hist, _ = np.histogram(second_closest_distances, bins=bins)
    second_closest_hist = second_closest_hist / np.sum(second_closest_hist, dtype=float)

    fig, axs = plt.subplots(1, 2, tight_layout=True, figsize=(14, 6))
    axs[0].hist(
        same_idx_distances,
        bins=bins,
        alpha=0.5,
        label="Same index",
        histtype="stepfilled",
    )
    axs[0].hist(
        second_closest_distances,
        bins=bins,
        alpha=0.5,
        label="Second closest",
        histtype="stepfilled",
    )

    plt.xticks(np.arange(np.min(bins), np.max(bins) + 1, 50.0))

    axs[1].plot(bins[:-1], np.cumsum(same_idx_hist))
    axs[1].plot(bins[:-1], np.cumsum(second_closest_hist))

    # Add axis labels
    axs[0].set_xlabel("Distance in pixels")
    axs[1].set_xlabel("Distance in pixels")
    axs[0].set_ylabel("Density")

    # Fix ticks
    plt.xticks(np.arange(np.min(bins), np.max(bins) + 1, 50.0))

    # Add a legend
    axs[0].legend(("Same Idx", "Closest different Idx"), loc="upper right")
    axs[1].legend(("Same Idx", "Closest different Idx"), loc="lower right")
    plt.title("Distance distribution between consecutive cuts.")
    plt.savefig("distributions.pdf")
