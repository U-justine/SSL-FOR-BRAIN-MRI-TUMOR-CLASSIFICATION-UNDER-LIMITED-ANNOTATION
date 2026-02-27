"""
Configuration Module for Brain MRI Tumor Classification
Complete GPU-optimized settings
"""

import os
import torch
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List
from pathlib import Path


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


@dataclass
class Config:
    """Complete configuration for Brain MRI classification."""
    
    # Paths
    data_root: str = "dataset"
    train_dir: str = "dataset/Training"
    test_dir: str = "dataset/Testing"
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    
    # Dataset
    num_classes: int = 4
    class_names: List[str] = field(default_factory=lambda: [
        "glioma_tumor", "meningioma_tumor", "pituitary_tumor", "no_tumor"
    ])
    image_size: int = 224
    label_percentages: List[float] = field(default_factory=lambda: [1.0, 0.10, 0.05, 0.01])
    
    # Model
    backbone: str = "resnet18"
    projection_dim: int = 128
    
    # Training - Supervised
    supervised_epochs: int = 15
    supervised_batch_size: int = 32
    supervised_lr: float = 3e-4
    
    # Training - SSL
    ssl_epochs_v1: int = 50
    ssl_epochs_v2: int = 100
    ssl_batch_size: int = 128
    ssl_lr: float = 3e-4
    ssl_temperature: float = 0.5
    
    # Training - Fine-tuning
    finetune_epochs: int = 15
    finetune_batch_size: int = 32
    finetune_lr: float = 1e-4
    freeze_backbone: bool = True
    
    # Optimization
    weight_decay: float = 1e-4
    momentum: float = 0.9
    gradient_clip_norm: float = 1.0
    use_scheduler: bool = True
    
    # GPU
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    use_amp: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    
    # Logging
    log_interval: int = 10
    save_interval: int = 5
    
    # Reproducibility
    seed: int = 42
    
    def __post_init__(self):
        set_seed(self.seed)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        if self.device == "cuda":
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  Using CPU (GPU not available)")
            self.use_amp = False
    
    def get_checkpoint_path(self, model_name: str, label_pct: float) -> str:
        return os.path.join(self.checkpoint_dir, f"{model_name}_{int(label_pct*100)}pct_best.pth")
