import torch
import torchvision
from torchmetrics import R2Score
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics import JaccardIndex
from dataset import MyDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="checkpoints/my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers,
):
    train_ds=MyDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader=DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        shuffle=True,
    )

    val_ds=MyDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader=DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
    )

    return train_loader, val_loader

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
            dice_score += (2*(preds*y).sum()) / ((preds+y).sum() + 1e-12)
    
    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

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
        torchvision.utils.save_image(x, f"{folder}/rgb_{idx+1}.png")
    model.train()