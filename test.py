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

TYPE_DATASET=1      #0 for binary, 1 for regression
#VAL_IMG_DIR="data/dataBinary/test_images/" if TYPE_DATASET==0 else "data/LANDMARK50/test_images"
#VAL_MASK_DIR="data/dataBinary/test_masks/" if TYPE_DATASET==0 else "data/LANDMARK50/test_masks"
#%VAL_IMG_DIR="data/dataBinary/test_images/" if TYPE_DATASET==0 else "data/Catello_verdura/TRANSFER_LEARNING/FILTERED/test_rgb_filter/"
#%VAL_MASK_DIR="data/dataBinary/test_masks/" if TYPE_DATASET==0 else "data/Catello_verdura/TRANSFER_LEARNING/FILTERED/test_ndvi_filter/"
VAL_IMG_DIR="data/dataBinary/test_images/" if TYPE_DATASET==0 else "../../AIIA_CIRO_RISO/NDVI/IMGS/"
VAL_MASK_DIR="data/dataBinary/test_masks/" if TYPE_DATASET==0 else "../../AIIA_CIRO_RISO/NDVI/MASKS/"

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    val_dir,
    val_maskdir,
    #batch_size,
    val_transform,
    num_workers,
):
    
    val_ds=MyDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader=DataLoader(
        val_ds,
        #batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
    )

    return val_loader

def check_binary_accuracy(loader, model, device):
    num_correct=0
    num_pixels=0
    dice_score=0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x=x.to(device)
            y=y.to(device).unsqueeze(1)
            preds=torch.sigmoid(model(x))
            preds=(preds>0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2*(preds*y).sum()) / ((preds+y).sum() + 1e-8)
    
    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    
    print(f"Dice score: {dice_score/len(loader)}")
    

def check_accuracy_per_batch(loader, model, device): #R-squared score and SSIM per batch
    r2score=R2Score().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)
    SSIM_array = []
    model.eval()

    with torch.no_grad():
        for idx, (x,y) in enumerate(loader):
            x=x.to(device)
            y=y.to(device).unsqueeze(1)
            preds=model(x)
            r2=r2score(preds.reshape(torch.numel(preds)),y.reshape(torch.numel(y)))
            SSIM_array.append(ssim(preds,y))
            print(f"The Accuracy for the batch n.{idx+1} is: SSIM={SSIM_array[idx]}; R2={r2.item()}")
        print(f"The mean SSIM score is {sum(SSIM_array)/len(SSIM_array)}")
    model.train()    

def check_R2score(loader, model, device): #R-squared: coefficient of determination
    r2score=R2Score().to(device)
    preds1=torch.Tensor([]).to(device)
    target1=torch.Tensor([]).to(device)
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x=x.to(device)
            y=y.to(device).unsqueeze(1)
            preds=model(x)
            target1=torch.cat((target1,y), 1)
            preds1=torch.cat((preds1,preds), 1)
        r2=r2score(preds1.reshape(torch.numel(preds1)),target1.reshape(torch.numel(target1)))
    print(f"The R-squared score is {r2.item()}")
    model.train()
        

def save_predictions_as_imgs(loader, model, folder, type_data, device):
    model.eval()
    print("=> Saving predictions")
    for idx, (x,y) in enumerate(loader):
        x=x.to(device)
        with torch.no_grad():
            if type_data==0:    #binary segmentation
                preds=torch.sigmoid(model(x))
                preds=(preds>0.5).float()
            else:               #regression
                preds=model(x)
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx+1}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/true_{idx+1}.png")
        torchvision.utils.save_image(x, f"{folder}/RGB_{idx+1}.png")
    
    #model.train()

def main():
    
    val_transform=A.Compose(
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
    #loss_fn=nn.BCEWithLogitsLoss() if TYPE_DATASET==0 else nn.MSELoss()
    #optimizer=optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    #scaler=torch.cuda.amp.GradScaler()

    val_loader = get_loaders(
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        #BATCH_SIZE,
        val_transform,
        NUM_WORKERS,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("checkpoints/my_checkpoint.pth.tar"), model)
        #load_checkpoint(torch.load("RESULTS/Luciano_checkpoints_V-net/my_checkpoint.pth.tar"), model)
        if TYPE_DATASET==0:
            check_binary_accuracy(val_loader, model, device=DEVICE)
    
    #loss_values = []
    i=0
    #%folder="saves/TEST_Performance_TRANSFER"
    folder="../../AIIA_CIRO_RISO/NDVI/pred_NDVI"
    #folder="../paper/predict/saves"
    if not os.path.exists(folder): 
        os.makedirs(folder)
    else:
        i=int(folder[10])
        i=i+1 
        folder = folder+str(i)
        os.makedirs(folder)

    
    #for epoch in range(NUM_EPOCHS):
        #print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        #running_loss=train_fn(train_loader, model, optimizer, loss_fn, scaler)
        #loss_values.append(running_loss)

        #check binary accuracy
    if TYPE_DATASET==0:
        check_binary_accuracy(val_loader, model, device=DEVICE)
        
        #print the output into a folder
    save_predictions_as_imgs(val_loader, model, folder, TYPE_DATASET, device=DEVICE)

        #save checkpoint model
        #checkpoint={"state_dict": model.state_dict(), "optimizer":optimizer.state_dict()}
        #save_checkpoint(checkpoint)
        
        #save loss function plot
        #plt.plot(loss_values)
        #plt.savefig(f"{folder}/_loss_fn.jpg")

    #check regression accuracy
    if TYPE_DATASET==1:
        check_accuracy_per_batch(val_loader, model, device=DEVICE) #R-squared score and SSIM per batch
        check_R2score(val_loader, model, device=DEVICE) #R-squared score
    
    

if __name__ == "__main__":
    main()