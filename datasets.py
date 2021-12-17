import torch
import torchvision
import numpy as np
from PIL import Image
from readlif.reader import LifFile
from torch.utils.data import Dataset
from typing import Optional
from pathlib import Path
import utils
import re


class ChloroplastsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: Path,
        dataset_file: Path,
        mask_type: str = "chloroplast",
        transforms: Optional[torchvision.transforms.Compose] = None,
    ):

        self.images_dir = data_dir / "images"
        self.lif_dir = data_dir / "lif"
        self.mask_type = mask_type
        self.mask_dir = data_dir / "masks" / mask_type

        with open(dataset_file, "r") as f:
            cell_prefixes = f.readlines()
        cell_prefixes = [x.strip() for x in cell_prefixes]

        self.images_list = sorted(
            [
                item
                for item in self.images_dir.iterdir()
                if item.suffix == ".tif"
                and item.name[: item.name.find("_")] in cell_prefixes
            ]
        )
        self.masks_list = sorted(
            [
                item
                for item in self.mask_dir.iterdir()
                if item.suffix == ".png"
                and item.name[: item.name.find("_")] in cell_prefixes
            ]
        )

        self.transforms = transforms

    def __getitem__(self, idx):
        lif_file_id, cell_id, plane_idx = re.split(" - |_", self.images_list[idx].stem)

        lif_file = LifFile(self.lif_dir.joinpath(f"{lif_file_id}.lif"))
        cell_lif_idx = next(
            (
                i
                for i, item in enumerate(lif_file.image_list)
                if item["name"] == cell_id
            ),
            None,
        )
        cell_lif_image = lif_file.get_image(cell_lif_idx)
        scale = cell_lif_image.scale
        pixel_width = 1.0 / scale[0]  # um/px
        pixel_height = 1.0 / scale[1]
        pixel_depth = 0.988
        voxel_density = pixel_depth * pixel_width * pixel_height

        tmp = [
            np.array(channel_image, dtype=np.float32)
            for channel_image in cell_lif_image.get_iter_c(t=0, z=int(plane_idx))
        ]
        # tmp.append(tmp[-1]) #simulate 3 channels?
        img = torch.from_numpy(np.array(tmp, dtype=np.float32) / 255.0)

        mask_filename = self.mask_dir.joinpath(
            f"{lif_file_id} - {cell_id}_{plane_idx}.png"
        )

        mask = Image.open(mask_filename)
        mask = np.array(mask)

        # instances are encoded as different colors
        # first id (=0) is the background, so remove it
        obj_ids = np.unique(mask)[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_obj = len(obj_ids)
        boxes = []
        for i in range(num_obj):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # there is only one class
        labels = torch.ones((num_obj,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        voxel_density = torch.as_tensor(voxel_density, dtype=torch.float32)

        image_id = torch.tensor([idx])
        try:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        except:
            print("Error trying to calculate the area of the bounding box.")
            exit()
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_obj,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["voxel_density"] = voxel_density
        # target["filename"] = self.images_list[idx]

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images_list)


if __name__ == "__main__":
    data_dir = Path("/home/pedro/repos/chloro_count/data")
    dataset_file = Path("/home/pedro/repos/chloro_count/datasets/train.txt")
    mask_type = "bundle_sheath"  # "chloroplast"

    dataset = ChloroplastsDataset(
        data_dir=data_dir, dataset_file=dataset_file, mask_type=mask_type
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=3,
        shuffle=False,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )
