
import os
import random
import numpy as np
import pandas as pd
import cv2
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torch.cuda.amp import autocast, GradScaler
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─── Cell 2: Reproducibility ───────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)

# ─── Cell 3: Dataset paths & class check ───────────────────────────
dataset_path = "/content/dataset"           # ← change if needed
train_path = os.path.join(dataset_path, "Training")
test_path  = os.path.join(dataset_path, "Testing")

def get_class_info(path):
    classes = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
    info = {}
    for cls in classes:
        count = len([f for f in os.listdir(os.path.join(path, cls)) if f.lower().endswith(('.jpg','.jpeg','.png'))])
        info[cls] = count
    return info

train_info = get_class_info(train_path)
test_info  = get_class_info(test_path)

print("Training set:", {k:v for k,v in train_info.items()}, f"Total: {sum(train_info.values())}")
print("Testing set: ", {k:v for k,v in test_info.items()},  f"Total: {sum(test_info.values())}")
class_names = list(train_info.keys())
num_classes = len(class_names)
print(f"Classes: {class_names} ({num_classes})")

# ─── Cell 4: Transforms ─────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# SimCLR-style strong augmentations (two views)
class SimCLRTransform:
    def __init__(self):
        self.base = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __call__(self, x):
        return self.base(x), self.base(x)

# ─── Cell 5: Datasets & loaders ────────────────────────────────────
full_train_dataset = ImageFolder(train_path, transform=train_transform)
test_dataset      = ImageFolder(test_path,  transform=test_transform)

# Stratified split → train / val
indices = list(range(len(full_train_dataset)))
labels  = [full_train_dataset.targets[i] for i in indices]
train_idx, val_idx = train_test_split(
    indices, test_size=0.2, stratify=labels, random_state=SEED
)

train_subset = Subset(full_train_dataset, train_idx)
val_subset   = Subset(full_train_dataset, val_idx)

print(f"Train: {len(train_subset)} | Val: {len(val_subset)} | Test: {len(test_dataset)}")

batch_size = 64
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)
val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)

# ─── Cell 6: SimCLR models & loss ──────────────────────────────────
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=512, hidden=2048, out=128, layers=2):
        super().__init__()
        seq = []
        prev = in_dim
        for _ in range(layers):
            seq += [nn.Linear(prev, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True)]
            prev = hidden
        seq.append(nn.Linear(prev, out))
        self.net = nn.Sequential(*seq)
    
    def forward(self, x):
        return self.net(x)

class SimCLRv1(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.encoder = models.resnet18(weights=None)
        self.encoder.fc = nn.Identity()
        self.head = ProjectionHead(512, 2048, proj_dim, 2)
    
    def forward(self, x):
        return self.head(self.encoder(x))

class SimCLRv2(nn.Module):
    def __init__(self, proj_dim=256):
        super().__init__()
        self.encoder = models.resnet18(weights=None)
        self.encoder.fc = nn.Identity()
        self.head = ProjectionHead(512, 4096, proj_dim, 3)
    
    def forward(self, x):
        return self.head(self.encoder(x))

def nt_xent_loss(z1, z2, temp=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.T) / temp
    mask = torch.eye(2*z1.size(0), device=z.device).bool()
    sim.masked_fill_(mask, -9e15)
    labels = torch.cat([torch.arange(z1.size(0), 2*z1.size(0)), torch.arange(0, z1.size(0))]).to(z.device)
    return F.cross_entropy(sim, labels)

# ─── Cell 7: Training functions ────────────────────────────────────
def train_ssl(model, loader, epochs=50, lr=3e-4, desc="SimCLR"):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    losses = []
    
    for epoch in range(epochs):
        model.train()
        total = 0.0
        loop = tqdm(loader, desc=f"{desc} {epoch+1}/{epochs}", leave=False)
        for x1, x2 in loop:
            x1, x2 = x1.to(device), x2.to(device)
            optimizer.zero_grad()
            with autocast():
                z1 = model(x1)
                z2 = model(x2)
                loss = nt_xent_loss(z1, z2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
        avg = total / len(loader)
        losses.append(avg)
        if (epoch+1) % 10 == 0:
            print(f"[{desc}] Epoch {epoch+1} loss: {avg:.4f}")
    return model, losses

def train_supervised(model, train_loader, val_loader, epochs=30, lr=1e-4, patience=7):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()
    
    best_acc = 0
    best_state = None
    patience_cnt = 0
    
    for epoch in range(epochs):
        model.train()
        correct, total_loss = 0, 0
        for img, lbl in tqdm(train_loader, desc=f"Train {epoch+1}", leave=False):
            img, lbl = img.to(device), lbl.to(device)
            optimizer.zero_grad()
            with autocast():
                out = model(img)
                loss = criterion(out, lbl)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            correct += (out.argmax(1) == lbl).sum().item()
        
        train_acc = 100 * correct / len(train_loader.dataset)
        
        model.eval()
        val_correct = 0
        with torch.no_grad(), autocast():
            for img, lbl in val_loader:
                img, lbl = img.to(device), lbl.to(device)
                out = model(img)
                val_correct += (out.argmax(1) == lbl).sum().item()
        
        val_acc = 100 * val_correct / len(val_loader.dataset)
        print(f"Epoch {epoch+1:2d} | Train acc {train_acc:.2f}% | Val acc {val_acc:.2f}%")
        
        scheduler.step()
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print("Early stopping")
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def evaluate(model, loader, names):
    model.eval()
    preds, trues = [], []
    with torch.no_grad(), autocast():
        for img, lbl in loader:
            img = img.to(device)
            out = model(img)
            preds.extend(out.argmax(1).cpu().numpy())
            trues.extend(lbl.numpy())
    
    acc = accuracy_score(trues, preds) * 100
    p, r, f1, _ = precision_recall_fscore_support(trues, preds, average='weighted')
    cm = confusion_matrix(trues, preds)
    
    print(classification_report(trues, preds, target_names=names, digits=3))
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "confusion_matrix": cm}

# ─── Cell 8: Run SimCLR pretraining (you can comment out if already done) ──
# ssl_ds = ImageFolder(train_path, transform=SimCLRTransform())
# ssl_loader = DataLoader(ssl_ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

# print("Training SimCLR v1...")
# simclr_v1 = SimCLRv1(128)
# simclr_v1, _ = train_ssl(simclr_v1, ssl_loader, epochs=50)
# torch.save(simclr_v1.encoder.state_dict(), "simclr_v1.pth")

# print("Training SimCLR v2...")
# simclr_v2 = SimCLRv2(256)
# simclr_v2, _ = train_ssl(simclr_v2, ssl_loader, epochs=50)
# torch.save(simclr_v2.encoder.state_dict(), "simclr_v2.pth")

# ─── Cell 9: Main experiment loop ──────────────────────────────────
label_fractions = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]
results = []

for frac in label_fractions:
    print(f"\n=== Running {int(frac*100)}% labeled data ===")
    
    # Subsample train set
    n_samples = int(len(train_subset) * frac)
    sub_idx = random.sample(range(len(train_subset)), n_samples)
    sub_train = Subset(train_subset, sub_idx)
    sub_loader = DataLoader(sub_train, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    
    # 1. From scratch
    print("From scratch...")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, num_classes)
    train_supervised(model, sub_loader, val_loader)
    metrics = evaluate(model, test_loader, class_names)
    results.append({'fraction': frac, 'method': 'scratch', **metrics})
    
    # 2. ImageNet
    print("ImageNet...")
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(512, num_classes)
    train_supervised(model, sub_loader, val_loader)
    metrics = evaluate(model, test_loader, class_names)
    results.append({'fraction': frac, 'method': 'imagenet', **metrics})
    
    # 3. SimCLR v1
    print("SimCLR v1...")
    enc = models.resnet18(weights=None)
    enc.fc = nn.Identity()
    enc.load_state_dict(torch.load("simclr_v1.pth"))
    model = nn.Sequential(enc, nn.Linear(512, num_classes))
    train_supervised(model, sub_loader, val_loader)
    metrics = evaluate(model, test_loader, class_names)
    results.append({'fraction': frac, 'method': 'simclr_v1', **metrics})
    
    # 4. SimCLR v2
    print("SimCLR v2...")
    enc = models.resnet18(weights=None)
    enc.fc = nn.Identity()
    enc.load_state_dict(torch.load("simclr_v2.pth"))
    model = nn.Sequential(enc, nn.Linear(512, num_classes))
    train_supervised(model, sub_loader, val_loader)
    metrics = evaluate(model, test_loader, class_names)
    results.append({'fraction': frac, 'method': 'simclr_v2', **metrics})

# ─── Cell 10: Results summary ──────────────────────────────────────
df = pd.DataFrame(results)
print(df.pivot_table(index='fraction', columns='method', values='accuracy'))

# Optional: save
df.to_csv("brain_tumor_results.csv", index=False)