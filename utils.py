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
        f"./checkpoints/{run_name}.pt",
    )


def load_run(model, optimizer, run_name):
    checkpoint = torch.load(f"./checkpoints/{run_name}.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer, checkpoint["epoch"]


def show_img_chw(img):
    plt.imshow(img.permute(1, 2, 0))
