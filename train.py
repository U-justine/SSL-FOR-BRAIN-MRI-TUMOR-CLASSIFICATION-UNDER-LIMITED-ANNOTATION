"""
Training Module - GPU Optimized
Complete training functions for supervised, SSL, and fine-tuning
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


def train_supervised(model, train_loader, test_loader, config, model_name, label_pct):
    """Train supervised model."""
    print(f"\n{'='*80}")
    print(f"TRAINING SUPERVISED: {model_name} @ {label_pct*100}% labels")
    print(f"{'='*80}")
    
    model = model.to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.supervised_lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.supervised_epochs) if config.use_scheduler else None
    scaler = GradScaler() if config.use_amp else None
    
    best_acc = 0.0
    
    for epoch in range(config.supervised_epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.supervised_epochs}")
        for images, labels in pbar:
            images, labels = images.to(config.device, non_blocking=True), labels.to(config.device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            if config.use_amp:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{train_loss/(pbar.n+1):.4f}', 'acc': f'{100.*correct/total:.2f}%'})
        
        if scheduler:
            scheduler.step()
        
        # Validation
        val_acc = validate(model, test_loader, config)
        print(f"Epoch {epoch+1}: Train Acc={100.*correct/total:.2f}%, Val Acc={val_acc:.2f}%")
        
        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, config.get_checkpoint_path(model_name, label_pct))
    
    print(f"✓ Best accuracy: {best_acc:.2f}%")
    return model


def train_ssl(model, ssl_loader, config, model_name, num_epochs):
    """Train SSL model with contrastive learning."""
    print(f"\n{'='*80}")
    print(f"SSL PRETRAINING: {model_name}")
    print(f"{'='*80}")
    
    model = model.to(config.device)
    criterion = nn.ModuleList([model.to(config.device) for _ in range(1)])[0]  # Move loss to device
    from model import NTXentLoss
    criterion = NTXentLoss(temperature=config.ssl_temperature)
    optimizer = Adam(model.parameters(), lr=config.ssl_lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs) if config.use_scheduler else None
    scaler = GradScaler() if config.use_amp else None
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(ssl_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for view1, view2 in pbar:
            view1, view2 = view1.to(config.device, non_blocking=True), view2.to(config.device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            if config.use_amp:
                with autocast():
                    _, z1 = model(view1)
                    _, z2 = model(view2)
                    loss = criterion(z1, z2)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                _, z1 = model(view1)
                _, z2 = model(view2)
                loss = criterion(z1, z2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{train_loss/(pbar.n+1):.4f}'})
        
        if scheduler:
            scheduler.step()
        
        print(f"Epoch {epoch+1}: Loss={train_loss/len(ssl_loader):.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config.save_interval == 0:
            save_checkpoint(model, config.get_checkpoint_path(model_name, 1.0))
    
    print(f"✓ SSL pretraining complete")
    return model


def train_finetuned(model, train_loader, test_loader, config, model_name, label_pct):
    """Fine-tune SSL model."""
    print(f"\n{'='*80}")
    print(f"FINE-TUNING: {model_name} @ {label_pct*100}% labels")
    print(f"{'='*80}")
    
    model = model.to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.finetune_lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.finetune_epochs) if config.use_scheduler else None
    scaler = GradScaler() if config.use_amp else None
    
    best_acc = 0.0
    
    for epoch in range(config.finetune_epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.finetune_epochs}")
        for images, labels in pbar:
            images, labels = images.to(config.device, non_blocking=True), labels.to(config.device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            if config.use_amp:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{train_loss/(pbar.n+1):.4f}', 'acc': f'{100.*correct/total:.2f}%'})
        
        if scheduler:
            scheduler.step()
        
        # Validation
        val_acc = validate(model, test_loader, config)
        print(f"Epoch {epoch+1}: Train Acc={100.*correct/total:.2f}%, Val Acc={val_acc:.2f}%")
        
        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, config.get_checkpoint_path(model_name, label_pct))
    
    print(f"✓ Best accuracy: {best_acc:.2f}%")
    return model


@torch.no_grad()
def validate(model, test_loader, config):
    """Validate model and return accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    for images, labels in test_loader:
        images, labels = images.to(config.device, non_blocking=True), labels.to(config.device, non_blocking=True)
        
        if config.use_amp:
            with autocast():
                outputs = model(images)
        else:
            outputs = model(images)
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total


def save_checkpoint(model, path):
    """Save model checkpoint."""
    if isinstance(model, nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    
    torch.save(state_dict, path)
    print(f"  ✓ Saved: {path}")


def load_checkpoint(model, path, device):
    """Load model checkpoint."""
    state_dict = torch.load(path, map_location=device)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    print(f"  ✓ Loaded: {path}")
    return model
