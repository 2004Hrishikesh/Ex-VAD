"""
Temporal-Enhanced ExVAD Model (Fixed Version)
Integrates temporal modeling with CMAF for improved video anomaly detection.

This model extends the Enhanced ExVAD with:
1. Per-frame feature extraction
2. Temporal Transformer/LSTM for sequence modeling
3. Motion encoding for dynamic patterns
4. Fused temporal + cross-modal features
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
from PIL import Image
import numpy as np

from src.models.enhanced_exvad_model import EnhancedExVADModel
from .temporal_modeling import (
    TemporalTransformer,
    TemporalLSTM,
    MotionEncoder,
    TemporalAnomalyDetector
)


class TemporalEnhancedExVAD(nn.Module):
    """
    Temporal-Enhanced ExVAD Model (Fixed Dimensions)
    
    Combines:
    - CMAF (Cross-Modal Attention Fusion) from EnhancedExVADModel
    - Temporal modeling for frame sequence understanding
    - Motion encoding for dynamic pattern detection
    """
    
    def __init__(self,
                 device='cuda',
                 temporal_hidden_dim=256,  # Reduced for memory
                 use_temporal_transformer=True,
                 use_temporal_lstm=True,
                 use_motion_encoding=True,
                 temporal_weight=0.3):
        """
        Args:
            device: Device to run model on
            temporal_hidden_dim: Hidden dimension for temporal modules
            use_temporal_transformer: Whether to use temporal transformer
            use_temporal_lstm: Whether to use temporal LSTM
            use_motion_encoding: Whether to use motion encoding
            temporal_weight: Weight for temporal score in final prediction
        """
        super().__init__()
        
        self.device = device
        self.temporal_weight = temporal_weight
        self.clip_feature_dim = 512  # CLIP ViT-B/32 output dimension
        self.temporal_hidden_dim = temporal_hidden_dim
        
        # Base CMAF model
        print("Initializing Temporal-Enhanced ExVAD Model...")
        print("Loading base CMAF model...")
        self.base_model = EnhancedExVADModel(device=device)
        
        # Temporal modeling module
        print("Initializing temporal modeling components...")
        self.temporal_detector = TemporalAnomalyDetector(
            input_dim=self.clip_feature_dim,
            hidden_dim=temporal_hidden_dim,
            use_transformer=use_temporal_transformer,
            use_lstm=use_temporal_lstm,
            use_motion=use_motion_encoding,
            dropout=0.2
        )
        
        # Simple fusion: just use temporal features directly for anomaly scoring
        # This avoids the complex CMAF feature extraction which caused dimension issues
        self.temporal_anomaly_head = nn.Sequential(
            nn.Linear(temporal_hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Move to device
        self.to(device)
        print(f"Temporal-Enhanced ExVAD initialized on {device}")
        
    def _freeze_base_model(self):
        """Freeze base CMAF model parameters"""
        for param in self.base_model.parameters():
            param.requires_grad = False
        print("Base CMAF model frozen")
    
    def _unfreeze_base_model(self):
        """Unfreeze base CMAF model for fine-tuning"""
        for param in self.base_model.parameters():
            param.requires_grad = True
        print("Base CMAF model unfrozen for fine-tuning")
    
    def extract_per_frame_features(self, frames: List[Image.Image]) -> torch.Tensor:
        """
        Extract CLIP features for each frame individually.
        
        Args:
            frames: List of PIL Images
            
        Returns:
            frame_features: (1, num_frames, feature_dim) tensor
        """
        frame_features_list = []
        
        # Get CLIP model and preprocessor from base model
        clip_model = self.base_model.madm.clip_model
        clip_preprocess = self.base_model.madm.clip_preprocess
        
        for frame in frames:
            try:
                # Convert to PIL Image if needed
                if torch.is_tensor(frame):
                    frame_np = frame.cpu().numpy()
                    if frame_np.shape[0] == 3:  # Channel first
                        frame_np = np.transpose(frame_np, (1, 2, 0))
                    if frame_np.dtype != np.uint8:
                        frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)
                    frame = Image.fromarray(frame_np)
                elif isinstance(frame, np.ndarray):
                    if frame.shape[0] == 3:
                        frame = np.transpose(frame, (1, 2, 0))
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).clip(0, 255).astype(np.uint8)
                    frame = Image.fromarray(frame)
                
                if not isinstance(frame, Image.Image):
                    frame = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
                
                if frame.mode != 'RGB':
                    frame = frame.convert('RGB')
                
                # Preprocess and extract CLIP features
                frame_tensor = clip_preprocess(frame).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    features = clip_model.encode_image(frame_tensor)
                    features = features.float()  # Convert from fp16
                    features = F.normalize(features, p=2, dim=1)
                
                frame_features_list.append(features)
                
            except Exception as e:
                # Fallback: zero features
                frame_features_list.append(torch.zeros(1, self.clip_feature_dim).to(self.device))
        
        # Stack all frame features: (num_frames, feature_dim) -> (1, num_frames, feature_dim)
        frame_features = torch.cat(frame_features_list, dim=0)  # (num_frames, 512)
        frame_features = frame_features.unsqueeze(0)  # (1, num_frames, 512)
        
        return frame_features
    
    def forward(self, 
                frames: Union[List[Image.Image], torch.Tensor],
                return_details=False) -> Tuple[torch.Tensor, str, str]:
        """
        Forward pass with temporal modeling.
        
        Args:
            frames: List of PIL Images or tensor
            return_details: Whether to return detailed outputs
            
        Returns:
            anomaly_score: Final anomaly probability
            explanation: Text explanation
            captions: Frame captions
        """
        # Get base CMAF outputs (for explanation and captions)
        with torch.no_grad():
            base_output = self.base_model(frames)
        
        if isinstance(base_output, tuple):
            base_score = base_output[0]
            explanation = base_output[1] if len(base_output) > 1 else "N/A"
            captions = base_output[2] if len(base_output) > 2 else "N/A"
        else:
            base_score = base_output
            explanation = "N/A"
            captions = "N/A"
        
        # Extract per-frame features for temporal modeling
        frame_features = self.extract_per_frame_features(frames)
        
        # Temporal modeling - get temporal features
        temporal_score, temporal_features, components = self.temporal_detector(
            frame_features, return_components=True
        )
        
        # Use temporal features for additional anomaly scoring
        temporal_anomaly = self.temporal_anomaly_head(temporal_features)
        
        # Weighted combination: base CMAF score + temporal score
        # base_score shape: (1,) or scalar
        # temporal_anomaly shape: (1, 1)
        
        if base_score.dim() == 0:
            base_score = base_score.unsqueeze(0)
        if base_score.dim() == 1:
            base_score = base_score.unsqueeze(1)
        
        combined_score = (1 - self.temporal_weight) * base_score + self.temporal_weight * temporal_anomaly
        
        # Enhanced explanation with temporal context
        enhanced_explanation = self._enhance_explanation(
            explanation, temporal_score.item(), components
        )
        
        if return_details:
            return combined_score, enhanced_explanation, captions, {
                'base_score': base_score,
                'temporal_score': temporal_score,
                'temporal_anomaly': temporal_anomaly,
                'temporal_components': components
            }
        
        return combined_score, enhanced_explanation, captions
    
    def _enhance_explanation(self, 
                            base_explanation: str,
                            temporal_score: float,
                            components: dict) -> str:
        """Enhance explanation with temporal context"""
        temporal_context = ""
        
        if temporal_score > 0.7:
            temporal_context = "Strong temporal anomaly pattern detected. "
        elif temporal_score > 0.4:
            temporal_context = "Moderate temporal irregularity observed. "
        else:
            temporal_context = "Normal temporal progression. "
        
        enhanced = temporal_context + str(base_explanation)
        return enhanced
    
    def get_temporal_attention(self, frames: List[Image.Image]) -> torch.Tensor:
        """
        Get temporal attention weights showing which frames are most important.
        """
        frame_features = self.extract_per_frame_features(frames)
        
        if hasattr(self.temporal_detector, 'temporal_transformer') and \
           self.temporal_detector.temporal_transformer is not None:
            _, _, attn_weights = self.temporal_detector.temporal_transformer(
                frame_features, return_attention=True
            )
            return attn_weights.squeeze(0)
        
        # Fallback: uniform attention
        return torch.ones(len(frames)) / len(frames)


def test_temporal_enhanced_model():
    """Test the temporal-enhanced model"""
    print("\n" + "="*70)
    print("Testing Temporal-Enhanced ExVAD Model")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    print("\nInitializing model...")
    model = TemporalEnhancedExVAD(
        device=device,
        temporal_hidden_dim=256,
        use_temporal_transformer=True,
        use_temporal_lstm=True,
        use_motion_encoding=True
    )
    
    model.eval()
    
    # Create dummy frames
    print("\nCreating dummy input (16 frames)...")
    dummy_frames = [Image.new('RGB', (224, 224), color=(128, 128, 128)) for _ in range(16)]
    
    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        score, explanation, captions, details = model(dummy_frames, return_details=True)
    
    print(f"\nResults:")
    print(f"   Combined Score: {score.item():.4f}")
    print(f"   Base Score: {details['base_score'].item():.4f}")
    print(f"   Temporal Score: {details['temporal_score'].item():.4f}")
    print(f"   Temporal Anomaly: {details['temporal_anomaly'].item():.4f}")
    print(f"   Explanation: {explanation[:100]}...")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameters:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    
    print("\nTemporal-Enhanced Model test complete!")


if __name__ == "__main__":
    test_temporal_enhanced_model()
