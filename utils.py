import torch
import os
import matplotlib.pyplot as plt
import copy
import random


def save_run(model, optimizer, run_name, epoch, model_name=None):
    try:
        os.mkdir("./checkpoints")
    except:
        pass
    try:
        os.mkdir(f"./checkpoints/{run_name}")
    except:
        pass
    if model_name is None:
        model_name = run_name
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        f"./checkpoints/{run_name}/{model_name}_epoch_{epoch}.pt",
    )


def load_run(model, optimizer, run_name, epoch):
    checkpoint = torch.load(f"./checkpoints/{run_name}/{run_name}_epoch_{epoch}.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer, checkpoint["epoch"]


def show_img_chw(img, denormalize=False):
    img = get_image_chw(img, denormalize)
    plt.imshow(img)


def get_image_chw(img, denormalize=False):
    # Scaling factors for each channel
    if denormalize:
        means = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        stds = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        img = img * stds
        img = img + means
    img = img.permute(1, 2, 0)
    return img


def get_experts_classes(N_SPLITS):
    with open("datasets/ucf101/test_train_splits/ucfTrainTestlist/testlist01.txt", "r") as file:
        lines_test = file.readlines()

    with open("datasets/ucf101/test_train_splits/ucfTrainTestlist/trainlist01.txt", "r") as file:
        lines_train = file.readlines()

    classes = list(set([int(line.strip().split(" ")[1]) for line in lines_train]))
    classes = [c - 1 for c in classes]
    n_classes = len(classes)
    splits = []

    for split_nr in range(N_SPLITS):
        if split_nr == (N_SPLITS - 1):
            split_classes = copy.deepcopy(classes)
        else:
            split_classes = random.sample(classes, k=(n_classes // N_SPLITS))
        # split_train = [line for line in lines_train if int(line.strip().split(" ")[1]) in split_classes]
        # split_test = [line for line in lines_test if int(line.strip().split(" ")[1]) in split_classes]
        # splits.append({"train": split_train, "test": split_test, "classes": split_classes})
        splits.append(split_classes)
        for c in split_classes:
            classes.remove(c)
    return splits
