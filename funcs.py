import tqdm
import torch
from utils import save_run
from metrics import accuracy
import matplotlib.pyplot as plt


def train_epoch(model, device, training_loader, epoch, optimizer, loss, label_smoothing, mixup, scheduler, config, wandb):
      model.train()
      print(f"Current epoch: {epoch}")
      curr_epoch_losses_train = []
      total_epoch_samples = 0
      total_epoch_correct_train = 0
      total_epoch_correct_top5_train = 0
      for i, data in enumerate(tqdm.tqdm(training_loader)):
            images, _, labels = data
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            if config["mixup"]:
                images, labels_mixed_up = mixup(images, labels)
                outputs = model(images).logits
                loss_train = loss(outputs, labels_mixed_up, label_smoothing=label_smoothing)
            else:
                outputs = model(images).logits
                loss_train = loss(outputs, labels, label_smoothing=label_smoothing)
                
            loss_train.backward()

            # logging
            total_epoch_samples += len(images)
            corrects = accuracy(outputs, labels, topk=(1, 5))
            total_epoch_correct_train += corrects[0]
            total_epoch_correct_top5_train += corrects[1]
            curr_epoch_losses_train.append(loss_train.item())
            wandb.log({"train_accuracy_step_top1": corrects[0] / len(images),
                       "train_accuracy_step_top5": corrects[1] / len(images),
                       "train_loss_step": loss_train.item()})

            if config["cosine_schedule"]:
                scheduler.step()
                wandb.log({"lr": float(scheduler.get_last_lr()[-1])})
            optimizer.step()

      wandb.log(
      {
            "train_accuracy_epoch_top1": total_epoch_correct_train / total_epoch_samples,
            "train_accuracy_epoch_top5": total_epoch_correct_top5_train / total_epoch_samples,
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
    total_correct_top5 = 0
    total_samples = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(validation_loader)):
            images, _, labels = data
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

    wandb.log({"valid_accuracy_epoch_top1": total_correct / total_samples,
               "valid_accuracy_epoch_top5": total_correct_top5 / total_samples,
               "valid_loss_epoch": total_loss / total_batches})


def test_epoch(model, test_loader, device, wandb):
    model.eval()
    total_correct = 0
    total_correct_top5 = 0
    total_samples = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(test_loader)):
            images = data["video"].squeeze(dim=0).to(device)
            label = data["label"].to(device)

            outputs = model(images).logits
            outputs = torch.sum(outputs, 0)

            outputs = outputs.unsqueeze(dim=0)
            label = label.unsqueeze(dim=0)
            corrects = accuracy(outputs, label, topk=(1, 5))
            total_correct += corrects[0]
            total_correct_top5 += corrects[1]
            total_samples += 1

    wandb.log({
         "test_accuracy_epoch_top1": total_correct / total_samples,
         "test_accuracy_epoch_top5": total_correct_top5 / total_samples
               })
