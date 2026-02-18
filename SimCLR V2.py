import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================= HYPERPARAMETERS =================
batch_size = 32
pretrain_epochs = 40
finetune_epochs = 20
learning_rate_pretrain = 3e-4
learning_rate_finetune = 1e-4
temperature = 0.5
num_classes = 2

# ================= TRANSFORMS =================
simclr_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ================= DATA =================
train_dir = "dataset/train"
val_dir = "dataset/val"

train_dataset = datasets.ImageFolder(train_dir, transform=eval_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=4)

print("Classes:", train_dataset.classes)

# ================= SIMCLR v2 MODEL =================
class SimCLRv2(nn.Module):
    def __init__(self, projection_dim=256):
        super().__init__()
        self.encoder = models.resnet18(pretrained=True)
        self.encoder.fc = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z

# ================= NT-XENT LOSS =================
def nt_xent_loss(z1, z2, temperature):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)

    similarity = torch.matmul(z, z.T)
    mask = torch.eye(2 * batch_size).bool().to(device)
    similarity = similarity[~mask].view(2 * batch_size, -1)

    positives = torch.sum(z1 * z2, dim=1)
    positives = torch.cat([positives, positives], dim=0)

    logits = similarity / temperature
    labels = torch.zeros(2 * batch_size).long().to(device)

    return nn.CrossEntropyLoss()(logits, labels)

# ================= PRETRAINING =================
simclr_model = SimCLRv2().to(device)
optimizer = optim.Adam(simclr_model.parameters(), lr=learning_rate_pretrain)

print("\nStarting SimCLR v2 Pretraining...\n")

for epoch in range(pretrain_epochs):
    simclr_model.train()
    total_loss = 0.0

    for images, _ in train_loader:
        images = images.to(device)

        x1 = torch.stack([simclr_transform(img.cpu()) for img in images]).to(device)
        x2 = torch.stack([simclr_transform(img.cpu()) for img in images]).to(device)

        _, z1 = simclr_model(x1)
        _, z2 = simclr_model(x2)

        loss = nt_xent_loss(z1, z2, temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Pretrain Epoch [{epoch+1}/{pretrain_epochs}] Loss: {total_loss/len(train_loader):.4f}")

# ================= CLASSIFIER =================
class SimCLRClassifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        return self.fc(x)

classifier = SimCLRClassifier(simclr_model.encoder, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=learning_rate_finetune)

# ================= FINE-TUNING =================
print("\nStarting Fine-Tuning...\n")

for epoch in range(finetune_epochs):
    classifier.train()
    correct = 0
    total = 0
    train_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = classifier(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total

    classifier.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = classifier(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total

    print(f"Epoch [{epoch+1}/{finetune_epochs}] "
          f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

# ================= SAVE MODEL =================
os.makedirs("saved_models", exist_ok=True)
torch.save(classifier.state_dict(), "saved_models/simclr_v2_brain_tumor.pth")

print("\nSimCLR v2 model saved successfully!")
