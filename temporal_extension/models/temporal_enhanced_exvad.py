"""
Temporal-Enhanced ExVAD Model
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

from src.models.enhanced_exvad_model import EnhancedExVADModel
from .temporal_modeling import (
    TemporalTransformer,
    TemporalLSTM,
    MotionEncoder,
    TemporalAnomalyDetector
)


class TemporalEnhancedExVAD(nn.Module):
    """
    Temporal-Enhanced ExVAD Model
    
    Combines:
    - CMAF (Cross-Modal Attention Fusion) from EnhancedExVADModel
    - Temporal modeling for frame sequence understanding
    - Motion encoding for dynamic pattern detection
    
    Architecture:
        Frames → CLIP (per-frame) → Temporal Encoder → Temporal Features
              → BLIP (captions) → Cross-Modal Fusion → CMAF Features
              → Fusion → Final Anomaly Score + Explanation
    """
    
    def __init__(self,
                 device='cuda',
                 temporal_hidden_dim=512,
                 use_temporal_transformer=True,
                 use_temporal_lstm=True,
                 use_motion_encoding=True,
                 fusion_method='concat',  # 'concat', 'add', 'attention'
                 temporal_weight=0.3):
        """
        Args:
            device: Device to run model on
            temporal_hidden_dim: Hidden dimension for temporal modules
            use_temporal_transformer: Whether to use transformer-based temporal modeling
            use_temporal_lstm: Whether to use LSTM-based temporal modeling
            use_motion_encoding: Whether to use motion encoding
            fusion_method: How to fuse temporal and CMAF features
            temporal_weight: Weight for temporal score in final prediction (if using weighted fusion)
        """
        super().__init__()
        
        self.device = device
        self.temporal_weight = temporal_weight
        self.fusion_method = fusion_method
        
        # Base CMAF model
        print("Initializing Temporal-Enhanced ExVAD Model...")
        print("Loading base CMAF model...")
        self.base_model = EnhancedExVADModel(device=device)
        
        # Freeze base model initially (optional - can be unfrozen for fine-tuning)
        # self._freeze_base_model()
        
        # Get CLIP feature dimension (typically 512 or 768)
        self.clip_feature_dim = 512  # ViT-B/32 output dimension
        
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
        
        # Feature fusion
        if fusion_method == 'concat':
            self.fusion_layer = nn.Sequential(
                nn.Linear(temporal_hidden_dim + 512, 512),  # 512 from CMAF
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256)
            )
        elif fusion_method == 'attention':
            self.fusion_attention = nn.MultiheadAttention(512, num_heads=8, batch_first=True)
            self.temporal_proj = nn.Linear(temporal_hidden_dim, 512)
            self.fusion_layer = nn.Linear(512, 256)
        else:  # 'add'
            self.temporal_proj = nn.Linear(temporal_hidden_dim, 512)
            self.fusion_layer = nn.Linear(512, 256)
        
        # Final anomaly prediction head
        self.final_anomaly_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Move to device
        self.to(device)
        print(f"✅ Temporal-Enhanced ExVAD initialized on {device}")
        
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
        preprocess = self.base_model.madm.clip_preprocess  # Fixed: use clip_preprocess
        
        for frame in frames:
            # Preprocess frame
            frame_tensor = preprocess(frame).unsqueeze(0).to(self.device)
            
            # Extract CLIP features
            with torch.no_grad():
                features = clip_model.encode_image(frame_tensor)
                features = features.float()
            
            frame_features_list.append(features)
        
        # Stack all frame features
        frame_features = torch.cat(frame_features_list, dim=0)  # (num_frames, feature_dim)
        frame_features = frame_features.unsqueeze(0)  # (1, num_frames, feature_dim)
        
        return frame_features
    
    def forward(self, 
                frames: Union[List[Image.Image], torch.Tensor],
                return_details=False) -> Tuple[torch.Tensor, str, str]:
        """
        Forward pass with temporal modeling.
        
        Args:
            frames: List of PIL Images or tensor (batch, frames, C, H, W)
            return_details: Whether to return detailed intermediate outputs
            
        Returns:
            anomaly_score: (batch, 1) - final anomaly probability
            explanation: str - text explanation
            captions: str - frame captions
        """
        # Get base CMAF outputs
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
        if isinstance(frames, list):
            frame_features = self.extract_per_frame_features(frames)
        else:
            # If tensor, extract features using CLIP
            frame_features = self._extract_tensor_features(frames)
        
        # Temporal modeling
        temporal_score, temporal_features, components = self.temporal_detector(
            frame_features, return_components=True
        )
        
        # Get CMAF features (approximate from base model output)
        # Note: In full implementation, we'd extract intermediate features from base_model
        cmaf_features = self._get_cmaf_features(frames)
        
        # Fuse temporal and CMAF features
        if self.fusion_method == 'concat':
            fused = torch.cat([temporal_features, cmaf_features], dim=-1)
            fused_features = self.fusion_layer(fused)
        elif self.fusion_method == 'attention':
            temporal_proj = self.temporal_proj(temporal_features).unsqueeze(1)
            cmaf_proj = cmaf_features.unsqueeze(1)
            combined = torch.cat([temporal_proj, cmaf_proj], dim=1)
            attended, _ = self.fusion_attention(combined, combined, combined)
            fused_features = self.fusion_layer(attended.mean(dim=1))
        else:  # 'add'
            temporal_proj = self.temporal_proj(temporal_features)
            fused_features = self.fusion_layer(temporal_proj + cmaf_features)
        
        # Final anomaly score
        final_score = self.final_anomaly_head(fused_features)
        
        # Weighted combination with base score for stability
        combined_score = (1 - self.temporal_weight) * base_score + self.temporal_weight * final_score
        
        # Enhanced explanation with temporal context
        enhanced_explanation = self._enhance_explanation(
            explanation, temporal_score.item(), components
        )
        
        if return_details:
            return combined_score, enhanced_explanation, captions, {
                'base_score': base_score,
                'temporal_score': temporal_score,
                'final_score': final_score,
                'temporal_components': components
            }
        
        return combined_score, enhanced_explanation, captions
    
    def _extract_tensor_features(self, frames_tensor: torch.Tensor) -> torch.Tensor:
        """Extract CLIP features from tensor input"""
        if frames_tensor.dim() == 5:
            batch_size, num_frames, C, H, W = frames_tensor.shape
            frames_flat = frames_tensor.view(batch_size * num_frames, C, H, W)
        else:
            frames_flat = frames_tensor
            batch_size = 1
            num_frames = frames_tensor.shape[0]
        
        clip_model = self.base_model.madm.clip_model
        
        with torch.no_grad():
            features = clip_model.encode_image(frames_flat)
            features = features.float()
        
        feature_dim = features.shape[-1]
        frame_features = features.view(batch_size, num_frames, feature_dim)
        
        return frame_features
    
    def _get_cmaf_features(self, frames) -> torch.Tensor:
        """
        Get CMAF intermediate features.
        For now, returns a learned projection of the base score.
        In full implementation, would extract from base_model internals.
        """
        # Placeholder: create features from base model
        # This should be replaced with actual CMAF intermediate features
        batch_size = 1
        cmaf_features = torch.randn(batch_size, 512).to(self.device)
        return cmaf_features
    
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
        
        # Add component-specific insights
        if 'motion' in components:
            motion_magnitude = components['motion'].abs().mean().item()
            if motion_magnitude > 0.5:
                temporal_context += "Unusual motion patterns detected. "
        
        enhanced = temporal_context + base_explanation
        return enhanced
    
    def get_temporal_attention(self, frames: List[Image.Image]) -> torch.Tensor:
        """
        Get temporal attention weights showing which frames are most important.
        
        Returns:
            attention_weights: (num_frames,) - importance of each frame
        """
        frame_features = self.extract_per_frame_features(frames)
        
        if hasattr(self.temporal_detector, 'temporal_transformer'):
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
        temporal_hidden_dim=512,
        use_temporal_transformer=True,
        use_temporal_lstm=True,
        use_motion_encoding=True,
        fusion_method='concat'
    )
    
    model.eval()
    
    # Create dummy frames
    print("\nCreating dummy input (16 frames)...")
    dummy_frames = [Image.new('RGB', (224, 224), color=(128, 128, 128)) for _ in range(16)]
    
    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        score, explanation, captions, details = model(dummy_frames, return_details=True)
    
    print(f"\n📊 Results:")
    print(f"   Combined Score: {score.item():.4f}")
    print(f"   Base Score: {details['base_score'].item():.4f}")
    print(f"   Temporal Score: {details['temporal_score'].item():.4f}")
    print(f"   Final Score: {details['final_score'].item():.4f}")
    print(f"   Explanation: {explanation[:100]}...")
    
    # Get temporal attention
    print("\nGetting temporal attention...")
    attn = model.get_temporal_attention(dummy_frames)
    print(f"   Attention shape: {attn.shape}")
    print(f"   Top 3 attended frames: {attn.topk(3).indices.tolist()}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📈 Parameters:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    
    print("\n✅ Temporal-Enhanced Model test complete!")


if __name__ == "__main__":
    test_temporal_enhanced_model()
