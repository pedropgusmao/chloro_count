from pathlib import Path
from os import PathLike
from PIL import Image
from stats import plot_img_with_ids
import numpy as np


def generate_segmented_dataset(
    image_folder: PathLike,
    segmentation_folder: PathLike,
    dataset_txt: PathLike,
    save_img_dataset,
):
    with open(dataset_txt, "r") as f:
        for cell_str in f:
            cell_str = cell_str.strip()
            num_imgs_in_this_cell = len(
                list(segmentation_folder.glob(f"**/{cell_str}*.png"))
            )
            uniq_ids = []
            for cut_idx in range(
                num_imgs_in_this_cell
            ):  # Some indices might be missing!!
                # Get unique
                seg_img = Image.open(
                    segmentation_folder / f"{cell_str}_{cut_idx}_chloroplast.png"
                )
                img_np = np.array(seg_img)
                uniq_ids.append(np.unique(img_np))

            uniq_ids = np.concatenate(uniq_ids)
            uniq_ids = np.unique(uniq_ids)
            for cut_idx in range(
                num_imgs_in_this_cell
            ):  # Some indices might be missing!!
                # Test image
                color_img = Image.open(image_folder / f"{cell_str}_{cut_idx}.tif")
                seg_img = Image.open(
                    segmentation_folder / f"{cell_str}_{cut_idx}_chloroplast.png"
                )

                img_np = np.array(seg_img)
                new_img = plot_img_with_ids(img_np, uniq_ids, color_img)
                new_img.save(save_img_dataset / f"{cell_str}_{cut_idx}.png")


if __name__ == "__main__":
    image_folder = Path("/home/pedro/repos/chloro_count/data/images")
    segmentation_folder = Path("/home/pedro/repos/chloro_count/data/masks/chloroplasts")
    save_dataset_root = Path("/home/pedro/repos/chloro_count/viz/original/chloroplasts")
    datasets = ["train", "val", "test"]
    for dataset in datasets:
        dataset_txt = (
            Path("/home/pedro/repos/chloro_count/data") / f"{dataset}_images.txt"
        )
        save_img_dataset = Path(save_dataset_root) / dataset
        generate_segmented_dataset(
            image_folder, segmentation_folder, dataset_txt, save_img_dataset
        )
