"""
Model Module - SimCLR and Classifiers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def get_backbone(backbone_name='resnet18'):
    """Get backbone network."""
    if backbone_name == 'resnet18':
        model = models.resnet18(weights=None)
        feat_dim = 512
    elif backbone_name == 'resnet34':
        model = models.resnet34(weights=None)
        feat_dim = 512
    elif backbone_name == 'resnet50':
        model = models.resnet50(weights=None)
        feat_dim = 2048
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    
    # Remove classification layer
    model = nn.Sequential(*list(model.children())[:-1])
    return model, feat_dim


class ProjectionHead(nn.Module):
    """MLP projection head for SimCLR."""
    
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class SimCLRv1(nn.Module):
    """SimCLR v1 with 2-layer projection head (512→256→128)."""
    
    def __init__(self, config):
        super().__init__()
        self.encoder, feat_dim = get_backbone(config.backbone)
        self.projection_head = ProjectionHead(feat_dim, [256], 128)
    
    def forward(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, 1)
        z = self.projection_head(h)
        return h, z


class SimCLRv2(nn.Module):
    """SimCLR v2 with 3-layer projection head (512→512→256→128)."""
    
    def __init__(self, config):
        super().__init__()
        self.encoder, feat_dim = get_backbone(config.backbone)
        self.projection_head = ProjectionHead(feat_dim, [512, 256], 128)
    
    def forward(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, 1)
        z = self.projection_head(h)
        return h, z


class SupervisedClassifier(nn.Module):
    """Supervised classifier from scratch."""
    
    def __init__(self, config):
        super().__init__()
        self.encoder, feat_dim = get_backbone(config.backbone)
        self.classifier = nn.Linear(feat_dim, config.num_classes)
    
    def forward(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, 1)
        return self.classifier(h)


class FineTunedClassifier(nn.Module):
    """Classifier built on pretrained SSL encoder."""
    
    def __init__(self, ssl_model, num_classes, freeze_encoder=True):
        super().__init__()
        self.encoder = ssl_model.encoder
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feat = self.encoder(dummy)
            feat_dim = torch.flatten(feat, 1).shape[1]
        
        self.classifier = nn.Linear(feat_dim, num_classes)
    
    def forward(self, x):
        with torch.set_grad_enabled(self.training and any(p.requires_grad for p in self.encoder.parameters())):
            h = self.encoder(x)
            h = torch.flatten(h, 1)
        return self.classifier(h)


class NTXentLoss(nn.Module):
    """NT-Xent contrastive loss for SimCLR."""
    
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1, z2):
        batch_size = z1.shape[0]
        device = z1.device
        
        # Normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Concatenate
        z = torch.cat([z1, z2], dim=0)
        
        # Similarity matrix
        sim_matrix = torch.mm(z, z.t()) / self.temperature
        
        # Create labels
        labels = torch.arange(batch_size, device=device)
        labels = torch.cat([labels + batch_size, labels])
        
        # Mask self-similarities
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(mask, -9e15)
        
        # Cross entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        return loss
