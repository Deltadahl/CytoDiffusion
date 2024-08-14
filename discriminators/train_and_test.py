import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import timm
import wandb
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import RandAugment
from tqdm import tqdm
from torch.distributions.beta import Beta
import gc
from random import Random
import torchvision.transforms.functional as TF
from sklearn.metrics import balanced_accuracy_score
import csv
import os
import pandas as pd

device = torch.device("cuda:0")
batch_size = 16
epochs = 50
n_runs = 1

model_names = ["efficientnetv2_rw_m", "vit_base_patch16_384"]
img_sizes = [360, 384]

experiment_name = 'Example'
wandb.init(project=f'Discriminator-{experiment_name}')

train_data_path = "../data/prepare_data/example_data_train_val_test/train"
val_data_path = "../data/prepare_data/example_data_train_val_test/val"
test_data_path = "../data/prepare_data/example_data_train_val_test/test"

def mixup_data(x, y, alpha=0.3, device=device):
    if alpha > 0:
        lam = Beta(alpha, alpha).sample().item()
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

mean_array_total = np.zeros(len(model_names))
mean_balanced_array_total = np.zeros(len(model_names))
std_array_total = np.zeros(len(model_names))
std_balanced_array_total = np.zeros(len(model_names))

for model_nr, (model_name, img_size) in enumerate(zip(model_names, img_sizes)):
    mean_array = np.zeros(n_runs)
    mean_balanced_array = np.zeros(n_runs)
    for dataset_nr in range(n_runs):
        seed_nr = model_nr * n_runs + dataset_nr
        random.seed(seed_nr)
        np.random.seed(seed_nr)
        torch.manual_seed(seed_nr)
        torch.cuda.manual_seed_all(seed_nr)

        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            RandAugment(),
            transforms.RandomRotation(360),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.125),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        val_test_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_dataset = datasets.ImageFolder(train_data_path, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_data_path, transform=val_test_transform)
        test_dataset = datasets.ImageFolder(test_data_path, transform=val_test_transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = timm.create_model(model_name, pretrained=True, num_classes=10)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)

        def train_model():
            val_not_improve_counter = 0
            best_accuracy = 0.0
            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                for tmp_idx, (images, labels) in tqdm(enumerate(train_loader)):
                    images, labels = images.to(device), labels.to(device)

                    images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=0.3)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                if (epoch + 1) % 1 == 0:
                    val_accuracy = evaluate_model()
                    print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}, Validation Accuracy: {val_accuracy}')
                    if val_accuracy >= best_accuracy:
                        val_not_improve_counter = 0
                        if not os.path.exists('saved_models'):
                            os.makedirs('saved_models')
                        best_accuracy = val_accuracy
                        torch.save(model.state_dict(), f'saved_models/{experiment_name}_{model_name}_{dataset_nr}.pth')
                        print(f'Saved model with accuracy: {best_accuracy}')
                    else:
                        val_not_improve_counter += 1
                        if val_not_improve_counter >= 15:
                            print(f'Early stopping at epoch {epoch+1}')
                            break

                    wandb.log({"epoch": epoch, "loss": running_loss / len(train_loader), "accuracy": val_accuracy})

        def evaluate_model():
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for tmp_idx, (images, labels) in enumerate(val_loader):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            return 100 * correct / total

        def test_model():
            all_labels = []
            all_predictions = []
            all_logit_differences = []
            all_image_names = []
            model.load_state_dict(torch.load(f'saved_models/{experiment_name}_{model_name}_{dataset_nr}.pth'))
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(test_loader):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)

                    top2_logits, top2_indices = torch.topk(outputs, 2, dim=1)
                    logit_differences = top2_logits[:, 0] - top2_logits[:, 1]

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())
                    all_logit_differences.extend(logit_differences.cpu().numpy())

                    # Fix: Calculate correct indices for the current batch
                    start_idx = batch_idx * batch_size
                    end_idx = start_idx + labels.size(0)
                    batch_image_names = [test_dataset.samples[i][0].split('/')[-1] for i in range(start_idx, end_idx)]
                    all_image_names.extend(batch_image_names)

            accuracy = 100 * correct / total
            balanced_acc = balanced_accuracy_score(all_labels, all_predictions) * 100
            print(f'Test Accuracy: {accuracy}')
            mean_array[dataset_nr] = accuracy
            mean_balanced_array[dataset_nr] = balanced_acc
            wandb.log({f"test_accuracy for {model_name}" : accuracy})
            wandb.log({f"test_balanced_accuracy for {model_name}" : balanced_acc})

            output_dir = experiment_name
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            output_file = os.path.join(output_dir, f'{experiment_name}_{model_name}_{dataset_nr}.csv')

            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Image_Name', 'Label', 'Prediction', 'Logit_Difference'])
                for name, label, pred, diff in zip(all_image_names, all_labels, all_predictions, all_logit_differences):
                    writer.writerow([name, label, pred, diff])

            print(f"Test results saved to {output_file}")

        train_model()
        test_model()

    mean_array_total[model_nr] = np.mean(mean_array)
    std_array_total[model_nr] = np.std(mean_array)
    mean_balanced_array_total[model_nr] = np.mean(mean_balanced_array)
    std_balanced_array_total[model_nr] = np.std(mean_balanced_array)

print(f"Mean: {mean_array_total}")
print(f"Std: {std_array_total}")
print(f"Mean Balanced: {mean_balanced_array_total}")
print(f"Std Balanced: {std_balanced_array_total}")
wandb.log({"mean": mean_array_total, "std": std_array_total})
wandb.log({"mean_balanced": mean_balanced_array_total, "std_balanced": std_balanced_array_total})
