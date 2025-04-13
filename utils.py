import torch
import os
import matplotlib.pyplot as plt


def save_run(model, optimizer, run_name, epoch):
    try:
        os.mkdir("./checkpoints")
    except:
        pass
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        f"./checkpoints/{run_name}_epoch_{epoch}.pt",
    )


def load_run(model, optimizer, run_name):
    checkpoint = torch.load(f"./checkpoints/{run_name}.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer, checkpoint["epoch"]


def show_img_chw(img, denormalize=False):
    # Scaling factors for each channel
    if denormalize:
        means = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        stds = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        img = img * stds
        img = img + means
    img = img.permute(1, 2, 0)
    plt.imshow(img)
