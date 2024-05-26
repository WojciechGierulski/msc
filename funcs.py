import tqdm
import torch
from utils import save_run


def train_epoch(model, device, training_loader, epoch, optimizer, loss, label_smoothing, wandb):
      model.train()
      print(f"Current epoch: {epoch}")
      curr_epoch_losses_train = []
      total_epoch_samples = 0
      total_epoch_correct_train = 0
      for i, data in enumerate(tqdm.tqdm(training_loader)):
            images, _, labels = data
            images = images.to(device)
            labels = labels.to(device)

            # images, labels_mixed_up = mixup(images, labels)

            optimizer.zero_grad()

            outputs = model(images).logits

            loss_train = loss(outputs, labels, label_smoothing=label_smoothing)
            loss_train.backward()

            # logging
            _, predicted = torch.max(outputs, dim=1)
            total_epoch_samples += len(images)
            correct = (predicted == labels).sum().item()
            total_epoch_correct_train += correct
            step_accuracy = correct / len(images)
            curr_epoch_losses_train.append(loss_train.item())
            wandb.log({"train_accuracy_step": step_accuracy, "train_loss_step": loss_train.item()})

      # scheduler.step()
      optimizer.step()

      wandb.log(
      {
            "train_accuracy_epoch": total_epoch_correct_train / total_epoch_samples,
            "train_loss_epoch": sum(curr_epoch_losses_train) / len(training_loader),
            "epoch_nr": epoch,
      }
      )
      save_run(model, optimizer, wandb.run.name, epoch)


def val_epoch(model, validation_loader, device, loss, wandb):
    model.eval()
    total_loss = 0
    total_batches = 0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(validation_loader)):
            images, _, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).logits

            loss_valid = loss(outputs, labels)

            # logging
            _, predicted = torch.max(outputs, dim=1)
            correct = (predicted == labels).sum().item()
            total_batches += 1
            total_loss += loss_valid.item()
            total_correct += correct
            total_samples += len(images)

    wandb.log({"valid_accuracy_epoch": total_correct / total_samples, "valid_loss_epoch": total_loss / total_batches})


def test_epoch(model, test_loader, device, wandb):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(test_loader)):
            images = data["video"].to(device)
            label = data["label"].to(device)

            outputs = model(images).logits
            outputs = torch.sum(outputs, 0)

            _, predicted = torch.max(outputs, dim=1)
            correct = 1 if predicted == label else 0
            total_correct += correct
            total_samples += 1

    wandb.log({"test_accuracy": total_correct / total_samples})
