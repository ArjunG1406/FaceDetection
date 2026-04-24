import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# GPU Setup
# ─────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")
if device.type == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

TRAIN_DIR = r"C:\Users\aravi\OneDrive\Desktop\Temp_project\train"
TEST_DIR  = r"C:\Users\aravi\OneDrive\Desktop\Temp_project\test"
IMG_SIZE  = 48
BATCH     = 64
EPOCHS    = 60
NUM_CLASS = 7

os.makedirs("models", exist_ok=True)

train_transforms = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
test_transforms = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
test_dataset  = datasets.ImageFolder(TEST_DIR,  transform=test_transforms)
train_loader  = DataLoader(train_dataset, batch_size=BATCH, shuffle=True,  num_workers=0, pin_memory=True)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)
print(f"\nClasses : {train_dataset.classes}")
print(f"Train   : {len(train_dataset)} images")
print(f"Test    : {len(test_dataset)} images\n")

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.BatchNorm1d(256), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

model = EmotionCNN(NUM_CLASS).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4, verbose=True)

def train_epoch():
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(train_loader), correct / total

def eval_epoch():
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.item()
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(test_loader), correct / total

best_val_acc = 0
patience_counter = 0
PATIENCE = 10
history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

print("=" * 70)
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_acc = train_epoch()
    vl_loss, vl_acc = eval_epoch()
    scheduler.step(vl_loss)
    elapsed = time.time() - t0

    history["train_acc"].append(tr_acc)
    history["val_acc"].append(vl_acc)
    history["train_loss"].append(tr_loss)
    history["val_loss"].append(vl_loss)

    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
          f"Val Loss: {vl_loss:.4f} Acc: {vl_acc:.4f} | "
          f"Time: {elapsed:.1f}s")

    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        torch.save(model.state_dict(), "models/emotion_model.pth")
        print(f"  💾 Best model saved! (val_acc={vl_acc:.4f})")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n⏹ Early stopping at epoch {epoch}")
            break

print(f"\n✅ Done! Best val accuracy: {best_val_acc:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history["train_acc"], label="Train")
axes[0].plot(history["val_acc"],   label="Val")
axes[0].set_title("Accuracy"); axes[0].legend()
axes[1].plot(history["train_loss"], label="Train")
axes[1].plot(history["val_loss"],   label="Val")
axes[1].set_title("Loss"); axes[1].legend()
plt.savefig("training_history.png")
plt.show()