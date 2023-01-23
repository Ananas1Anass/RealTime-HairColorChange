import torch
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision
import os
import numpy as np
from dataset import HairSegDataset
from torch.utils.data import DataLoader


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# parameters
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
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
        num_workers=4,
        pin_memory=True,
):
    train_ds = HairSegDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = HairSegDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def get_loader(
        val_dir,
        val_maskdir,
        batch_size,
        val_transform,
        num_workers=4,
        pin_memory=True,
):

    val_ds = HairSegDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return val_loader


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
#            dice_score += (2 * (preds * y).sum()) / (
  #                  (preds + y).sum() + 1e-8
   #         )
    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}"
    )
 #   print(f"Dice score: {dice_score / len(loader)}")
    model.train()
    


def save_predictions_as_imgs(
        loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        if idx % 30 == 0:
            x = x.to(device=device)
            with torch.no_grad():
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()
            torchvision.utils.save_image(
                preds, f"{folder}/pred_{idx}.png"
            )
            torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()


def predict_image(
        image, model, folder='test_images', device="cuda"
):
    model.eval()
    image = np.array(image.convert("RGB"))
    tfms = A.Compose(
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
    # unsqueeze provides the batch dimension
    x = tfms(image=image)["image"].to(device=device).unsqueeze(0)        # [[1], [2, [3]].squeeze(0)= [1,2,3]
    y = tfms(image=image)["image"]

    with torch.no_grad():
        preds = torch.sigmoid(model(x))
        preds = (preds > 0.5).float()

    #torchvision.utils.save_image(preds, f"{folder}/prediction.png")
    #torchvision.utils.save_image(y, f"{folder}/image.png")

    preds = (preds.squeeze().cpu().numpy().round())
    fig = plt.figure(figsize=(15, 7))
    fig.add_subplot(1, 2, 1)
    plt.imshow(A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH)(image=image)["image"])
    fig.add_subplot(1, 2, 2)
    plt.imshow(preds, cmap='gray', vmin=0, vmax=1)
    plt.show()
