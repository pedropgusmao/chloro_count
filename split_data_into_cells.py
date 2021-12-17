from pathlib import Path
import torch 
from datasets import ChloroplastsDataset
from train import get_instance_segmentation_model, get_transform
from tqdm import tqdm
import utils

def save_to_cell(data_folder, cell_folder, dataset_txt, dataset_name):
    with open(dataset_txt) as f:
        for cell_name in f:
            save_path = cell_folder / cell / 'segments.pt'
            torch.save(results, )



def main():

    # Use GPU if the CPU, if a GPU is not available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load the model
    num_classes = 2
    model = get_instance_segmentation_model(num_classes)
    model.load_state_dict(torch.load('saved_models/chloro_count.pth'))
    model.eval()
    model.to(device)

    # Save folders
    data_folder =  Path("/home/pedro/repos/chloro_count/data")
    dataset_cells_folder = Path("/home/pedro/repos/chloro_count/datasets/cells")
    masks_folder = data_folder / "masks" / "chloroplasts"


    for dataset_name in ["train", "val", "test"]:
        print(f"Processing {dataset_name} dataset.")
        with open(data_folder / f"{dataset_name}_images.txt", 'r') as f:
            for cell_name in f:
                cell_folder = dataset_cells_folder / dataset_name / cell_name.strip() 
                cell_folder.mkdir(parents=True, exist_ok=True)
                cell_file = cell_folder / "cell.txt"
                with open(cell_file, 'w') as fc:
                    fc.writelines([cell_name])

                dataset = ChloroplastsDataset(
                            dataset_dir = Path(data_folder),
                            dataset_file = Path(cell_file),
                            transforms = get_transform(train=False),
                        )

                dataloader = torch.utils.data.DataLoader(
                            dataset, batch_size=1,
                            shuffle=False,
                            num_workers=4,
                            collate_fn=utils.collate_fn,
                        )

                # Generate all masks:
                for idx, images_targets in enumerate(dataloader):
                    images, targets = images_targets
                    images = [image.to(device) for image in images]
                    prediction = model(images)

                    pred_masks = prediction[0]['masks']
                    pred_masks = pred_masks.to('cpu').detach().numpy()

                    pred_scores = prediction[0]['scores']
                    pred_scores = pred_scores.to('cpu').detach().numpy()

                    target_masks = targets[0]["masks"].to('cpu').detach().numpy()
                    results = {'pred':pred_masks, 'scores': pred_scores, 
                                'filename': dataset.images_list[idx],
                                'target':target_masks, 
                                'voxel_density': targets[0]['voxel_density']}
                    
                    real_idx = str(dataset.images_list[idx]).split('_')[-1].split('.')[0]
                    torch.save(results, cell_folder / f"cut_{real_idx}.pt")


if __name__ =='__main__':
    main()