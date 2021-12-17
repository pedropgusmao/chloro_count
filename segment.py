import torch
import torchvision
import numpy as np
import utils
from pathlib import Path
from train import get_transform, get_instance_segmentation_model
from datasets import ChloroplastsDataset
from PIL import Image
import matplotlib.pyplot as pl

from scripts.stats import plot_img_with_ids, get_idx_mapping


def target_to_matrix_seg(target_mat: np.ndarray):
    result = np.zeros(target_mat[0].shape, dtype=np.uint8)
    for idx, seg in enumerate(target_mat):
        result[seg > 0] = idx + 1

    return result


def get_closest(target, pred):
    max_ious = []
    areas = []
    for idx_t in np.unique(target):
        this_target_list_iou = []
        if idx_t < 1:
            continue
        component1 = target == idx_t
        areas.append(component1.sum())
        for idx_p in np.unique(pred):
            if idx_p < 1:
                continue
            component2 = pred == idx_p
            overlap = component1 * component2  # Logical AND
            union = component1 + component2  # Logical OR
            iou = overlap.sum() / float(union.sum())
            this_target_list_iou.append(iou)
        max_ious.append(np.max(this_target_list_iou))

    return max_ious, areas


def tensor_to_matrix_seg(seg_tensor: np.ndarray, threshold: float = 0.2):
    # (S, C, H, W)
    seg_matrix = np.zeros((seg_tensor[0][0].shape), dtype=np.uint8)  # (H, W)
    for idx, seg in enumerate(seg_tensor):
        seg_mono = np.squeeze(seg)
        seg_matrix[seg_mono > threshold] = (
            idx + 1
        )  # Avoid 0 = background #Avoid 0 = background

    return seg_matrix


def count_seg_from_matrix(seg_matrix: np.ndarray):
    num_seg_plus_bg = len(np.unique(seg_matrix))
    num_seg = num_seg_plus_bg - 1
    return num_seg


def get_pixel_area_from_seg_matrix(seg_matrix: np.ndarray, voxel_density: float = 1.0):
    return np.count_nonzero(seg_matrix) * voxel_density


def get_idx_from_previous(previous, current, threshold):
    max_idx = previous[-1][-1]
    new_current = []
    for current_center, old_idx in current:
        list_dist = []
        for prev_center, prev_idx in previous:
            list_dist.append(np.linalg.norm(current_center - prev_center))

        min_dist = np.amin(np.array(list_dist))
        if min_dist < threshold:
            prev_pos = np.argmin(np.array(list_dist))
            prev_idx = previous[prev_pos][-1]
            this_idx = prev_idx
        else:
            this_idx = max_idx + 1
            max_idx = this_idx
        new_current.append((current_center, this_idx))

    return new_current


def get_segment_centers(image):
    uniq_idx = np.unique(image)
    list_centers = []
    for idx in uniq_idx:
        if idx < 1:
            continue
        _x, _y = np.where(image[0] == idx)
        x_c = np.mean(_x, dtype=float)
        y_c = np.mean(_y, dtype=float)
        list_centers.append((np.array([x_c, y_c]), idx))
    return list_centers


def accumulate_segments(
    segments: np.ndarray, scores: np.ndarray, threshold: float = 0.8
):
    full_segmented = np.zeros_like(segments[0], dtype=np.uint8)
    for idx, score in enumerate(scores):
        if score > threshold:
            full_segmented[segments[idx]] = idx

    return full_segmented


def filter_on_score(predictions, scores, threshold):
    segmented_image = np.zeros_like(predictions[0][0], dtype=np.uint8)
    for idx, score in enumerate(scores):
        if score >= threshold:
            segmented_image[predictions[idx][0] > 0.5] = idx + 1

    return segmented_image


def update_indices(segmented_image, new_ids):
    old_ids = np.sort(np.unique(segmented_image))
    temp = np.zeros_like(segmented_image)
    for old_idx, new_idx in zip(old_ids, new_ids):
        temp[segmented_image == old_idx] = new_idx
    return temp


def main():

    # Save folders
    data_folder = Path("/home/pedro/repos/chloro_count/data")
    dataset_cells_folder = Path("/home/pedro/repos/chloro_count/datasets/cells")
    masks_folder = data_folder / "masks" / "bundle_sheath"

    # for dataset_name in ["train", "val", "test"]:
    gt_volumes = []
    pred_volumes = []
    ratio_volumes = []
    total_list_iou = []
    for dataset_name in ["val"]:
        print(f"Processing {dataset_name} dataset.")
        with open(data_folder / f"{dataset_name}_images.txt", "r") as f:
            target_cell_volumes = []
            pred_cell_volumes = []
            ratio_cell_volumes = []
            for cell_name in f:
                cell_name = cell_name.strip()
                cell_folder = dataset_cells_folder / dataset_name / cell_name
                num_cuts = len([x for x in cell_folder.glob("*.pt")])
                segment_centers_in_planes = []
                current_cell_pred_volume = 0
                current_cell_target_volume = 0

                for idx in range(num_cuts):
                    cut_file = cell_folder / f"cut_{idx}.pt"
                    results = torch.load(cut_file)

                    predictions = results["pred"]
                    voxel_density = results["voxel_density"]
                    target = results["target"]
                    scores = results["scores"]

                    ## Calculate area
                    squeezed_target = target_to_matrix_seg(target)
                    # squeezed_pred = tensor_to_matrix_seg(predictions)
                    squeezed_pred = filter_on_score(predictions, scores, threshold=0.0)

                    vol_target = get_pixel_area_from_seg_matrix(
                        squeezed_target, voxel_density=voxel_density
                    )
                    vol_pred = get_pixel_area_from_seg_matrix(
                        squeezed_pred, voxel_density=voxel_density
                    )
                    gt_volumes.append(vol_target)
                    pred_volumes.append(vol_pred)
                    ratio_volumes.append(vol_pred / vol_target)

                    # Update Cell value
                    current_cell_pred_volume += vol_pred
                    current_cell_target_volume += vol_target

                    list_iou, areas = get_closest(squeezed_target, squeezed_pred)
                    total_list_iou += list_iou

                    ##Plot regular segmentation
                    path_original_image = Path(results["filename"])
                    # print(path_original_image)
                    color_img = Image.open(path_original_image)

                    # Select scores
                    # if vol_pred/vol_target > 5:
                    # if 'Series003_1' in str(path_original_image):
                    if 0.0 in list_iou:
                        segmented_image = squeezed_pred
                        instances_ids = np.unique(segmented_image)

                        new_img = plot_img_with_ids(
                            segmented_image, instances_ids, color_img
                        )
                        new_img.save(f"{cell_name}_{idx}.png")
                target_cell_volumes.append(current_cell_target_volume)
                pred_cell_volumes.append(current_cell_pred_volume)
                ratio_cell_volumes.append(
                    current_cell_pred_volume / current_cell_target_volume
                )
    print(total_list_iou)

    # fig = pl.hist(total_list_iou, bins=np.linspace(0.0, 0.95, 18))
    # pl.title("IoU density for Recall")
    # pl.xlabel("IoU")
    # pl.ylabel("Frequency")
    # pl.savefig("abc.png")

    print("gt cell-wise")
    print(len(target_cell_volumes))
    print(np.mean(target_cell_volumes))
    print(np.std(target_cell_volumes))
    print("pred cell-wise")
    print(np.mean(pred_cell_volumes))
    print(np.std(pred_cell_volumes))
    print("Ratio cell-wise")
    print(np.mean(ratio_cell_volumes))
    print(np.std(ratio_cell_volumes))

    # Histogram over volumes
    fig, axs = pl.subplots(1, 2, sharey=True)
    axs[0].hist(target_cell_volumes, bins=np.linspace(0.0, 1000, 40))
    axs[0].set_title("Ground-truth")
    axs[0].set_xlabel("Chroloplast cell volume")
    axs[0].set_ylabel("Frequency")

    axs[1].hist(pred_cell_volumes, bins=np.linspace(0.0, 1000, 40))
    axs[1].set_title("Predicted")
    axs[1].set_xlabel("Chroloplast cell volume")
    axs[1].set_ylabel("Frequency")
    fig.savefig("total_chloroplast_volume_pred_per_cell.png")

    # Histogram over volumes
    fig, axs = pl.subplots(1, 2, sharey=True)
    axs[0].hist(gt_volumes, bins=np.linspace(0.0, 200, 40))
    axs[0].set_title("Ground-truth")
    axs[0].set_xlabel("Chroloplast section volume")
    axs[0].set_ylabel("Frequency")

    axs[1].hist(pred_volumes, bins=np.linspace(0.0, 200, 40))
    axs[1].set_title("Predicted")
    axs[1].set_xlabel("Chroloplast section volume")
    axs[1].set_ylabel("Frequency")
    fig.savefig("total_chloroplast_volume_pred_per_section.png")

    print("gt plane-wise")
    print(np.mean(gt_volumes))
    print(np.std(gt_volumes))
    print("pred plane-wise")
    print(np.mean(pred_volumes))
    print(np.std(pred_volumes))
    print("ratio plane-wise")
    print(np.mean(ratio_volumes))
    print(np.std(ratio_volumes))
    # print(segment_centers_in_planes)
    # exit()

    # pil_image = Image.fromarray(np.squeeze(segmented_image))
    # path_original_image = Path(results['filename'])
    # pil_image.save(f"{path_original_image.stem}_temp.png")


if __name__ == "__main__":
    main()