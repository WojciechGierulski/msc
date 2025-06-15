import tqdm
import torch
from metrics import accuracy
from utils import show_img_chw
from torch.nn.functional import softmax
from models.switch.utils import collect_loss, clear_memory, set_experts_param, set_grad_except_router


def train_epoch(
    model,
    device,
    training_loader,
    epoch,
    optimizer,
    loss,
    label_smoothing,
    mixup,
    scheduler,
    config,
    wandb,
):
    return
    optimizer.zero_grad()
    model.train()
    print(f"Current epoch: {epoch}")
    curr_epoch_losses_train = []
    curr_epoch_losses_classification_train = []
    curr_epoch_losses_load_balance_train = []
    total_epoch_samples = 0
    total_batches = 0
    total_epoch_correct_train = 0
    total_epoch_correct_top5_train = 0
    processed_videos = set()
    pbar = tqdm.tqdm(total=training_loader.dataset.num_videos)

    if (
        config["switch"]
        and config["router"]
        and config["pretrain_experts_epochs"]
        and config["freeze_net_iters"]
        and epoch == config.get("pretrain_experts_epochs", -100) + 1
    ):
        model = set_grad_except_router(model, config["switch_layers"], len(model.timesformer.encoder.layer), False)

    for iter_nr, data in enumerate(training_loader):
        if (
            config["switch"]
            and config["router"]
            and config["pretrain_experts_epochs"]
            and config["freeze_net_iters"]
            and epoch == config.get("pretrain_experts_epochs", -100) + 1
            and iter_nr == config.get("freeze_net_iters", 0)
        ):
            model = set_grad_except_router(model, config["switch_layers"], len(model.timesformer.encoder.layer), True)

        images = data["video"]
        labels = data["label"]

        images = images.to(device)
        labels = labels.to(device)

        if config["mixup"] and len(labels.unique()) > 1:
            images, labels_mixed_up = mixup(images.permute((0, 2, 1, 3, 4)), labels)
            images = images.permute((0, 2, 1, 3, 4))
            labels = labels_mixed_up.argmax(axis=1)
            if (
                config["switch"]
            ):
                model = set_experts_param(model, config["switch_layers"], "labels", labels)
                model = set_experts_param(model, config["switch_layers"], "video_indexes", data["video_index"])
            outputs = model(images).logits
            loss_train_classification = loss(outputs, labels_mixed_up)
        else:
            if (
                config["switch"]
            ):
                model = set_experts_param(model, config["switch_layers"], "labels", labels)
                model = set_experts_param(model, config["switch_layers"], "video_indexes", data["video_index"])
            outputs = model(images).logits
            loss_train_classification = loss(outputs, labels, label_smoothing=label_smoothing)

        if config["switch"] and config.get("pretrain_experts_epochs", 0) < epoch:
            load_balancing_loss = collect_loss(model, config["switch_layers"])
            load_balancing_loss = config["load_balance_alpha"] * config["n_experts"] * load_balancing_loss
        else:
            load_balancing_loss = torch.tensor(0).to(device)

        loss_train = loss_train_classification + load_balancing_loss

        if config["cosine_schedule"]:
            scheduler.step()
            wandb.log({"lr": float(scheduler.get_last_lr()[-1])})
        else:
            wandb.log({"lr": optimizer.param_groups[0]["lr"]})

        loss_train.backward()
        optimizer.step()
        optimizer.zero_grad()

        if config["switch"]:
            clear_memory(model, config["switch_layers"])

        # logging
        total_batches += 1
        total_epoch_samples += len(images)
        corrects = accuracy(outputs, labels, topk=(1, 5))
        total_epoch_correct_train += corrects[0]
        total_epoch_correct_top5_train += corrects[1]
        curr_epoch_losses_train.append(loss_train.item())
        curr_epoch_losses_classification_train.append(loss_train_classification.item())
        curr_epoch_losses_load_balance_train.append(load_balancing_loss.item())
        wandb.log(
            {
                "train_accuracy_step_top1": corrects[0] / len(images),
                "train_accuracy_step_top5": corrects[1] / len(images),
                "train_loss_step": loss_train.item(),
                "train_load_balancing_loss_step": load_balancing_loss.item(),
                "train_classification_loss_step": loss_train_classification.item(),
            }
        )
        [processed_videos.add(el.item()) for el in data["video_index"]]
        pbar.n = len(processed_videos)
        pbar.refresh()

    wandb.log(
        {
            "train_accuracy_epoch_top1": total_epoch_correct_train / total_epoch_samples,
            "train_accuracy_epoch_top5": total_epoch_correct_top5_train / total_epoch_samples,
            "train_loss_epoch": sum(curr_epoch_losses_train) / total_batches,
            "train_load_balancing_loss_epoch": sum(curr_epoch_losses_load_balance_train) / total_batches,
            "train_classification_loss_epoch": sum(curr_epoch_losses_classification_train) / total_batches,
            "epoch_nr": epoch,
        }
    )


def val_epoch(model, validation_loader, device, loss, wandb):
    model.eval()
    total_loss = 0
    total_batches = 0
    total_correct = 0
    total_correct_top5 = 0
    total_samples = 0
    processed_videos = set()
    with torch.no_grad():
        pbar = tqdm.tqdm(total=validation_loader.dataset.num_videos)
        for data in validation_loader:
            start = time.perf_counter()
            images = data["video"]
            labels = data["label"]
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).logits

            loss_valid = loss(outputs, labels)

            # logging
            corrects = accuracy(outputs, labels, topk=(1, 5))
            total_batches += 1
            total_loss += loss_valid.item()
            total_correct += corrects[0]
            total_correct_top5 += corrects[1]
            total_samples += len(images)
            [processed_videos.add(el.item()) for el in data["video_index"]]
            pbar.n = len(processed_videos)
            pbar.refresh()
            print(time.perf_counter() - start)

    wandb.log(
        {
            "valid_accuracy_epoch_top1": total_correct / total_samples,
            "valid_accuracy_epoch_top5": total_correct_top5 / total_samples,
            "valid_loss_epoch": total_loss / total_batches,
        }
    )
    return (total_correct / total_samples).item()


def test(model, test_loader, device, wandb):
    model.eval()
    total_correct = 0
    total_correct_top5 = 0
    total_samples = 0
    processed_videos = set()
    with torch.no_grad():
        pbar = tqdm.tqdm(total=test_loader.dataset.num_videos)
        for data in test_loader:
            images = data["video"]
            labels = data["label"]
            batch, clips_per_vid, frames, ch, w, h = images.shape
            images = images.reshape((batch * clips_per_vid, frames, ch, w, h))
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).logits
            classes = outputs.shape[-1]
            outputs = outputs.reshape((batch, clips_per_vid, classes)).mean(axis=1)

            # logging
            corrects = accuracy(outputs, labels, topk=(1, 5))
            total_correct += corrects[0]
            total_correct_top5 += corrects[1]
            total_samples += len(labels)
            [processed_videos.add(el.item()) for el in data["video_index"]]
            pbar.n = len(processed_videos)
            pbar.refresh()

    wandb.log(
        {
            "test_accuracy_epoch_top1": total_correct / total_samples,
            "test_accuracy_epoch_top5": total_correct_top5 / total_samples,
        }
    )
