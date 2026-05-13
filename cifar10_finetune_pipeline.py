import os
import copy
import json
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights, VGG16_Weights

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix


 #CONFIG #
SEED = 42
DATA_DIR = "./data"
OUTPUT_DIR = "./outputs"
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_cifar10_model.pth")
REPORT_JSON_PATH = os.path.join(OUTPUT_DIR, "best_result.json")
PLOT_PATH = os.path.join(OUTPUT_DIR, "training_curves.png")
NUM_CLASSES = 10
VAL_RATIO = 0.1
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# REPRODUCIBILITY

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)



# LABELS

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]



# DATA

def get_transforms(img_size: int = 224):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_transform, test_transform


class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


def get_dataloaders(batch_size: int, img_size: int = 224):
    train_transform, test_transform = get_transforms(img_size)

    full_train = datasets.CIFAR10(root=DATA_DIR, train=True, download=True)
    test_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_transform)

    train_len = int((1 - VAL_RATIO) * len(full_train))
    val_len = len(full_train) - train_len
    train_subset, val_subset = random_split(
        full_train,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_dataset = TransformedSubset(train_subset, transform=train_transform)
    val_dataset = TransformedSubset(val_subset, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, val_loader, test_loader



# MODELS

def build_model(model_name: str, freeze_backbone: bool = False):
    model_name = model_name.lower()

    if model_name == "resnet18":
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, NUM_CLASSES)
        )
    elif model_name == "vgg16":
        model = models.vgg16(weights=VGG16_Weights.DEFAULT)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, NUM_CLASSES)
    else:
        raise ValueError("Supported models: resnet18, vgg16")

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

        if model_name == "resnet18":
            for param in model.fc.parameters():
                param.requires_grad = True
        elif model_name == "vgg16":
            for param in model.classifier.parameters():
                param.requires_grad = True

    return model.to(DEVICE)



# METRICS

def compute_metrics(y_true, y_pred) -> Dict:
    acc = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
    }



# TRAIN / EVAL

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    preds_all, labels_all = [], []

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(labels_all, preds_all)
    return epoch_loss, metrics


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    preds_all, labels_all = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(labels_all, preds_all)
    return epoch_loss, metrics, labels_all, preds_all



# PLOTS

def plot_history(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train / Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train / Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



# EXPERIMENT

def run_experiment(model_name: str, lr: float, batch_size: int, epochs: int, weight_decay: float, freeze_backbone: bool):
    print("=" * 80)
    print(f"Running experiment: model={model_name}, lr={lr}, batch_size={batch_size}, epochs={epochs}, weight_decay={weight_decay}, freeze_backbone={freeze_backbone}")

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)
    model = build_model(model_name=model_name, freeze_backbone=freeze_backbone)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_metrics, _, _ = evaluate(model, val_loader, criterion)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)

    test_loss, test_metrics, y_true, y_pred = evaluate(model, test_loader, criterion)

    report = classification_report(y_true, y_pred, target_names=CIFAR10_CLASSES, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    result = {
        "model_name": model_name,
        "lr": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "weight_decay": weight_decay,
        "freeze_backbone": freeze_backbone,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "history": history,
        "state_dict": copy.deepcopy(model.state_dict())
    }

    return result


# =========================
# HYPERPARAM SEARCH
# =========================
def hyperparameter_search():
    search_space = [
        {"model_name": "resnet18", "lr": 1e-3, "batch_size": 64, "epochs": 10, "weight_decay": 1e-4, "freeze_backbone": False},
        {"model_name": "resnet18", "lr": 3e-4, "batch_size": 64, "epochs": 12, "weight_decay": 1e-4, "freeze_backbone": False},
        {"model_name": "resnet18", "lr": 1e-4, "batch_size": 128, "epochs": 15, "weight_decay": 5e-4, "freeze_backbone": False},
        {"model_name": "vgg16", "lr": 1e-4, "batch_size": 32, "epochs": 10, "weight_decay": 1e-4, "freeze_backbone": False},
        {"model_name": "vgg16", "lr": 3e-4, "batch_size": 32, "epochs": 12, "weight_decay": 5e-4, "freeze_backbone": False},
        {"model_name": "resnet18", "lr": 1e-3, "batch_size": 64, "epochs": 8, "weight_decay": 1e-4, "freeze_backbone": True},
        {"model_name": "vgg16", "lr": 1e-3, "batch_size": 32, "epochs": 8, "weight_decay": 1e-4, "freeze_backbone": True},
    ]

    all_results = []
    best_result = None

    for config in search_space:
        result = run_experiment(**config)
        all_results.append(result)

        if best_result is None or result["test_metrics"]["accuracy"] > best_result["test_metrics"]["accuracy"]:
            best_result = result

    return best_result, all_results


# =========================
# SAVE / LOAD
# =========================
def save_best_model(best_result):
    torch.save({
        "model_name": best_result["model_name"],
        "state_dict": best_result["state_dict"],
        "num_classes": NUM_CLASSES,
        "classes": CIFAR10_CLASSES,
        "test_metrics": best_result["test_metrics"],
        "best_val_acc": best_result["best_val_acc"],
        "config": {
            "lr": best_result["lr"],
            "batch_size": best_result["batch_size"],
            "epochs": best_result["epochs"],
            "weight_decay": best_result["weight_decay"],
            "freeze_backbone": best_result["freeze_backbone"]
        }
    }, BEST_MODEL_PATH)

    report_to_save = {
        "model_name": best_result["model_name"],
        "config": {
            "lr": best_result["lr"],
            "batch_size": best_result["batch_size"],
            "epochs": best_result["epochs"],
            "weight_decay": best_result["weight_decay"],
            "freeze_backbone": best_result["freeze_backbone"]
        },
        "best_val_acc": best_result["best_val_acc"],
        "test_loss": best_result["test_loss"],
        "test_metrics": best_result["test_metrics"],
        "classification_report": best_result["classification_report"],
        "confusion_matrix": best_result["confusion_matrix"]
    }

    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report_to_save, f, indent=4)

    plot_history(best_result["history"], PLOT_PATH)


def load_model_for_inference(model_path: str):
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model_name = checkpoint["model_name"]
    model = build_model(model_name=model_name, freeze_backbone=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["classes"]


def predict_image(image_path: str, model_path: str = BEST_MODEL_PATH):
    model, classes = load_model_for_inference(model_path)
    _, test_transform = get_transforms(img_size=224)

    image = Image.open(image_path).convert("RGB")
    image_tensor = test_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        predicted_class = torch.argmax(outputs, dim=1).item()

    return classes[predicted_class]


# =========================
# MAIN
# =========================
def main():
    start_time = time.time()

    best_result, all_results = hyperparameter_search()
    save_best_model(best_result)

    print("\nBest experiment:")
    print(f"Model: {best_result['model_name']}")
    print(f"Accuracy: {best_result['test_metrics']['accuracy']:.4f}")
    print(f"Precision (macro): {best_result['test_metrics']['precision_macro']:.4f}")
    print(f"Recall (macro): {best_result['test_metrics']['recall_macro']:.4f}")
    print(f"F1 (macro): {best_result['test_metrics']['f1_macro']:.4f}")
    print("\nClassification report:\n")
    print(best_result["classification_report"])
    print(f"\nSaved best model to: {BEST_MODEL_PATH}")
    print(f"Saved report to: {REPORT_JSON_PATH}")
    print(f"Saved plots to: {PLOT_PATH}")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed / 60:.2f} minutes")


if __name__ == "__main__":
    main()
