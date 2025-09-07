import torch
import torch.nn as nn
from .aegm import SimpleAEGM
from .madm import SimpleMADM

class ExVADModel(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        # Initialize components
        self.aegm = SimpleAEGM(device)
        self.madm = SimpleMADM(device)
        
        print(f"✅ ExVAD model initialized on {device}")
    
    def forward(self, frames):
        """Forward pass through the model"""
        # Generate explanation
        explanation, captions = self.aegm(frames)
        
        # Compute anomaly score
        anomaly_score = self.madm(frames)
        
        return anomaly_score, explanation, captions
    
    def to(self, device):
        """Move model to device"""
        self.device = device
        self.aegm = self.aegm.to(device)
        self.madm = self.madm.to(device)
        return super().to(device)
