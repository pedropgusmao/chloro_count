import numpy as np
import torch
from pathlib import Path
from PIL import Image
from readlif.reader import LifFile
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
from pyobb.obb import OBB
from volumes import Ellipsoid
from tqdm import tqdm
import pickle

# from ..volumes import Ellipsoid
def lineplot(ax, bb, idx1, idx2):
    ax.plot(
        [bb[idx1][0], bb[idx2][0]],
        [bb[idx1][1], bb[idx2][1]],
        [bb[idx1][2], bb[idx2][2]],
        color="blue",
        linewidth=0.5,
    )


def plot_cube(P: np.ndarray):
    # make 100 random points

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # plot points
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], color="g", marker="*", s=10)

    obb = OBB.build_from_points(P)
    bb = obb.points

    # ax.plot(BB[0:4])
    for idx in range(4):
        ax.plot(
            [bb[idx][0], bb[(idx + 1) % 4][0]],
            [bb[idx][1], bb[(idx + 1) % 4][1]],
            [bb[idx][2], bb[(idx + 1) % 4][2]],
            color="blue",
            linewidth=0.5,
        )
        lineplot(ax, bb, idx + 4, (4 + (idx + 1) % 4))
    # ax.axis("equal")
    lineplot(ax, bb, 0, 5)
    lineplot(ax, bb, 1, 4)
    lineplot(ax, bb, 2, 7)
    lineplot(ax, bb, 3, 6)

    # plot ellipsoid
    plt.show()
    plt.close(fig)
    del fig


def get_contour_from_np_images(images_np: np.ndarray, scale_matrix, chloro_id: int):
    """Returns points (in 3D metric) of the countour for each sementation plane
        for a given chloro_plast_id

    Args:
        images_np (np.ndarray): Planes containing segmentation.
        scale_matrix ([type]): Scale used to convert pixels to um.
        chloro_id (int): Id of a specific chloroplast in a cell

    Returns:
        [type]: [description]
    """
    P = []
    for idx, image_np in enumerate(images_np):
        this_chloroplast = np.where(image_np == chloro_id, image_np, 0)
        contours = find_boundaries(this_chloroplast, mode="inner")
        x, y = np.where(contours)
        these_points = np.array(
            [x, y, idx * np.ones_like(x)], dtype=np.float32
        ).transpose()
        P.append(these_points)

    P = np.concatenate(P)
    P = np.matmul(P, scale_matrix)
    return P


def calculate_chloroplast_volume():
    # Choose which image
    lif_name = "22A41#7.lif"
    image_name = "22A41#7 - Series003"
    img_lif = LifFile(lif_dir / lif_name)

    cell = img_lif.get_image()
    # Get scales
    x_px_um, y_px_um, z_px_um, _ = cell.scale
    scale_matrix = np.diag([1 / x_px_um, 1 / y_px_um, 1 / z_px_um])

    # Get points
    P = []
    for idx in range(11):
        img_pil = Image.open(cell_dir / f"{image_name}_{idx}_bundle_sheath.png")
        img_np = np.array(img_pil)
        contours = find_boundaries(img_np, mode="inner")
        x, y = np.where(contours)
        these_points = np.array(
            [x, y, idx * np.ones_like(x)], dtype=np.float32
        ).transpose()
        P.append(these_points)
    P = np.concatenate(P)
    P = np.matmul(P, scale_matrix)
    plot_cube(P)

    ## Get Chloroplast Volumes
    cells = sorted(
        list({str(x.name).rsplit("_")[0] for x in list(cell_dir.glob("**/*.png"))})
    )

    def images_to_ndarray(list_img_paths):
        return [np.array(Image.open(x)) for x in list_img_paths]

    for cell in cells:
        list_img_paths = sorted(list(chloro_dir.glob(f"**/{cell}*.png")))
        list_ndarrays = images_to_ndarray(list_img_paths)
        chloro_ids = np.unique(np.array(list_ndarrays))
        # chloro_ids.
        # for id in chloro_ids:
        #    points = [np.where(x == id) for x in list_ndarrays]
        #   print(points)


def get_canonical_volume(images_np, scale_matrix):

    total_volume = 0.0
    area_factor = scale_matrix[0, 0] * scale_matrix[1, 1]
    for idx, image_np in enumerate(images_np):
        total_volume += np.count_nonzero(image_np) * area_factor * scale_matrix[2, 2]
    return total_volume


def get_unique_ids(segmented_images_dir, image_name, list_planes):
    # Get unique IDs. Works for chloroplasts and bundle sheath
    unique_ids_list = []
    for img_path in list_planes:
        img_pil = Image.open(img_path)
        img_np = np.array(img_pil)
        unique_ids_list.append(np.unique(img_np))

    unique_ids = np.concatenate(unique_ids_list, axis=None)
    unique_ids = np.unique(unique_ids)
    unique_ids_no_bg = np.delete(unique_ids, np.where(unique_ids == 0))

    return unique_ids_no_bg


def get_factor_for_inner_ellipsoid(P, center, radii, rotation):
    Pc = np.dot(P - center, np.transpose(rotation))
    fractions = np.divide(np.multiply(Pc, Pc), np.multiply(radii, radii))
    fractions = np.sum(fractions, axis=1)
    inner_factor = np.sqrt(np.min(fractions))

    return inner_factor


def get_segments_for_this_id(segmented_images_dir, list_planes, image_name, chloro_id):
    images_np = []
    for plane_id in range(len(list_planes)):
        # This has to be in order.
        img_pil = Image.open(segmented_images_dir / f"{image_name}_{plane_id}.png")
        img_np = np.array(img_pil)
        img_np = np.where(img_np == chloro_id, chloro_id, 0)
        images_np.append(img_np)
    return images_np


def get_ellipsoid_volumes(images_np, scale_matrix, chloro_id, plot_ellipsoid=False):
    P = get_contour_from_np_images(images_np, scale_matrix, chloro_id=chloro_id)
    E = Ellipsoid()
    center, radii, rotation = E.get_params(P)

    # Get inner Ellipsoid
    inner_factor = get_factor_for_inner_ellipsoid(
        P=P, center=center, radii=radii, rotation=rotation
    )

    inner_volume = E.get_volume(radii * inner_factor)
    middle_volume = E.get_volume(radii * (1 + inner_factor) / 2.0)
    outter_volume = E.get_volume(radii)
    if plot_ellipsoid:
        E.plot(center, radii, rotation, P=P, inner_factor=inner_factor)

    return inner_volume, middle_volume, outter_volume


if __name__ == "__main__":
    # Define dataset
    lif_dir = Path("/home/pedro/repos/chloro_count/data/lif")

    # segmented_images_dir, dataset_name = select_dataset("bundle_sheath")
    dataset_name = "bundle_sheath"
    # dataset_name = "chloroplast"
    segmented_images_dir = Path(
        f"/home/pedro/repos/chloro_count/data/masks/{dataset_name}"
    )

    list_all_images = sorted(segmented_images_dir.glob("*.png"))
    list_all_plants = set([x.name.split(" - ")[0] for x in list_all_images])

    # Allocate memory:
    num_segmented_items = 0
    for plant_name in list_all_plants:
        lif_name = f"{plant_name}.lif"
        img_lif = LifFile(lif_dir / lif_name)

        list_this_plant = sorted(segmented_images_dir.glob(f"{plant_name}*.png"))
        list_series = set(
            [x.name.split(f"Series")[1].split("_")[0] for x in list_this_plant]
        )
        for series in tqdm(list_series):
            image_name = f"{plant_name} - Series{series}"
            # Get points
            list_planes = sorted(
                segmented_images_dir.glob(f"{plant_name} - Series{series}_*.png")
            )
            unique_ids_no_bg = get_unique_ids(
                segmented_images_dir, image_name, list_planes
            )
            num_segmented_items += len(unique_ids_no_bg)

    list_inner = np.zeros((num_segmented_items,), dtype=np.float32)
    list_mid = np.zeros_like(list_inner)
    list_out = np.zeros_like(list_inner)
    list_canonical = np.zeros_like(list_inner)

    item_count = 0
    error_count = 0
    for plant_name in list_all_plants:
        print(f"Processing {plant_name}")
        lif_name = f"{plant_name}.lif"
        img_lif = LifFile(lif_dir / lif_name)

        list_this_plant = sorted(segmented_images_dir.glob(f"{plant_name}*.png"))
        list_series = set(
            [x.name.split(f"Series")[1].split("_")[0] for x in list_this_plant]
        )
        for series in tqdm(list_series):
            image_name = f"{plant_name} - Series{series}"

            cell = img_lif.get_image()
            # Get scales
            x_px_um, y_px_um, z_px_um, _ = cell.scale
            scale_matrix = np.diag([1 / x_px_um, 1 / y_px_um, 1 / z_px_um])

            # Get points
            list_planes = sorted(
                segmented_images_dir.glob(f"{plant_name} - Series{series}_*.png")
            )
            print(list_planes)

            unique_ids_no_bg = get_unique_ids(
                segmented_images_dir, image_name, list_planes
            )

            # Get unique planes
            for seg_id in unique_ids_no_bg:
                images_np = get_segments_for_this_id(
                    segmented_images_dir, list_planes, image_name, chloro_id=seg_id
                )  # Should be one for bundle sheath
                print(len(images_np))

                # Get Ellipsoid Volume
                try:
                    inner_v, mid_v, out_v = get_ellipsoid_volumes(
                        images_np=images_np,
                        scale_matrix=scale_matrix,
                        chloro_id=seg_id,
                        plot_ellipsoid=False,
                    )

                    regular_volume = get_canonical_volume(
                        images_np, scale_matrix=scale_matrix
                    )

                    list_inner[item_count] = inner_v
                    list_mid[item_count] = mid_v
                    list_out[item_count] = out_v
                    list_canonical[item_count] = regular_volume
                except:
                    error_count += 1
                    print(
                        f"Found Error with image {plant_name} - Series{series} ID={seg_id}"
                    )

                item_count += 1

    print(f"Total errors: {error_count}")
    print("Saving...")
    a = {}
    a["inner"] = list_inner
    a["mid"] = list_mid
    a["out"] = list_out
    a["canonical"] = list_canonical
    with open(f"{dataset_name}_histograms.pickle", "wb") as f:
        pickle.dump(a, f)
