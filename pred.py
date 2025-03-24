import torch
import torchvision
from torchmetrics import R2Score
from torchmetrics.image import StructuralSimilarityIndexMeasure
from dataset import MyDataset
from torch.utils.data import DataLoader
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
from model import UNET


DEVICE="cuda" if torch.cuda.is_available() else "cpu"
#BATCH_SIZE=8
NUM_WORKERS=2
IMAGE_HEIGHT=340 #number of rows
IMAGE_WIDTH=450 #number of colums
LOAD_MODEL=True#False

### MODEL SELECTION
TYPE_DATASET=0      #0 for binary, 1 for regression
## Enter 0 for binary classification using VeGan dataset by Madec, S., et al. Sci Data 10, 302 (2023).
## Enter 1 for NDVI prediction using Vie-Net

## INPUT SELECTION
INPUT_IMG_DIR="../../AIIA_CIRO_RISO/NDVI/IMGS/" 
OUTPUT_FOLDER="../../AIIA_CIRO_RISO/NDVI/pred_NDVI2"
CHECKPOINT = "RESULTS\Luciano_checkpoints_V-net/my_checkpoint.pth.tar" # for Vie-Net
#CHECKPOINT = "saves/checkpoints/my_checkpoint_binary.pth.tar" # for bynary

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    pred_dir,
    pred_transform,
    num_workers,
):
    
    pred_ds=MyDataset(
        image_dir=pred_dir,
        transform=pred_transform,
    )

    pred_loader=DataLoader(
        pred_ds,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
    )

    return pred_loader




def save_predictions_as_imgs(loader, model, folder, type_data, device):
    model.eval()
    print("=> Saving predictions")
    for idx, (x) in enumerate(loader):
        x=x.to(device)
        with torch.no_grad():
            if type_data==0:    #binary segmentation
                preds=torch.sigmoid(model(x))
                preds=(preds>0.5).float()
            else:               #regression
                preds=model(x)
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx+1}.png")
        torchvision.utils.save_image(x, f"{folder}/RGB_{idx+1}.png")
    

def main():
    
    pred_transform=A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model=UNET(in_channels=3, out_channels=1).to(DEVICE)

    pred_loader = get_loaders(
        INPUT_IMG_DIR,
        pred_transform,
        NUM_WORKERS,
    )

    load_checkpoint(torch.load(CHECKPOINT), model)
    i=0
    folder=OUTPUT_FOLDER
    if not os.path.exists(folder): 
        os.makedirs(folder)
    else:
        i=int(folder[10])
        i=i+1 
        folder = folder+str(i)
        os.makedirs(folder)
    save_predictions_as_imgs(pred_loader, model, OUTPUT_FOLDER, TYPE_DATASET, device=DEVICE)
  

if __name__ == "__main__":
    main()
