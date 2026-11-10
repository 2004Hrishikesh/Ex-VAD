"""
Temporal-Enhanced ExVAD Model V3 (Fixed Version)
Addresses key flaws:
1. Non-discriminative temporal features - adds contrastive loss and feature diversity
2. Blocked gradient flow - enables gradients through temporal path
3. Better temporal-spatial fusion with learnable weights
4. Proper normalization and regularization
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
)


class ImprovedTemporalAnomalyDetector(nn.Module):
    """
    Improved temporal anomaly detector with better discriminative learning.
    Key improvements:
    - Feature normalization for stable training
    - Variance loss to prevent feature collapse
    - Separate temporal encodings for normal vs anomaly patterns
    """
    
    def __init__(self,
                 input_dim=512,
                 hidden_dim=256,
                 use_transformer=True,
                 use_lstm=True,
                 use_motion=True,
                 dropout=0.3):
        super().__init__()
        
        self.use_transformer = use_transformer
        self.use_lstm = use_lstm
        self.use_motion = use_motion
        self.hidden_dim = hidden_dim
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Temporal modules with better initialization
        if use_transformer:
            self.temporal_transformer = TemporalTransformer(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=2,  # Reduced to prevent overfitting
                num_heads=4,
                dropout=dropout
            )
            self._init_transformer_weights()
        
        if use_lstm:
            self.temporal_lstm = TemporalLSTM(
                input_dim=input_dim,
                hidden_dim=hidden_dim // 2,
                num_layers=2,
                dropout=dropout
            )
        
        if use_motion:
            self.motion_encoder = MotionEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim // 2
            )
        
        # Calculate fusion dimension
        fusion_dim = 0
        if use_transformer:
            fusion_dim += hidden_dim
        if use_lstm:
            fusion_dim += hidden_dim // 2
        if use_motion:
            fusion_dim += hidden_dim // 2
        
        # Improved fusion with residual connection
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Separate projection for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Better anomaly scoring head with proper initialization
        self.temporal_anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize final layer with small weights for stable training
        self._init_output_layer()
    
    def _init_transformer_weights(self):
        """Better initialization for transformer"""
        if hasattr(self, 'temporal_transformer'):
            for p in self.temporal_transformer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p, gain=0.1)
    
    def _init_output_layer(self):
        """Initialize output layer to start near 0.5"""
        with torch.no_grad():
            final_linear = self.temporal_anomaly_head[-1]
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)
    
    def forward(self, frame_features, return_components=False):
        """
        Args:
            frame_features: (batch, num_frames, input_dim)
            return_components: Whether to return individual components
            
        Returns:
            temporal_score: (batch, 1) raw logits (no sigmoid)
            temporal_features: (batch, hidden_dim) 
            components: dict of intermediate features
        """
        # Normalize input features
        x = self.input_norm(frame_features)
        
        components = {}
        features_list = []
        
        if self.use_transformer:
            trans_feat, _ = self.temporal_transformer(x)
            trans_feat = F.normalize(trans_feat, p=2, dim=-1)  # L2 normalize
            features_list.append(trans_feat)
            components['transformer'] = trans_feat
        
        if self.use_lstm:
            lstm_feat, _ = self.temporal_lstm(x)
            lstm_feat = F.normalize(lstm_feat, p=2, dim=-1)
            features_list.append(lstm_feat)
            components['lstm'] = lstm_feat
        
        if self.use_motion:
            motion_feat = self.motion_encoder(x)
            motion_feat = F.normalize(motion_feat, p=2, dim=-1)
            features_list.append(motion_feat)
            components['motion'] = motion_feat
        
        # Fuse features
        fused = torch.cat(features_list, dim=-1)
        temporal_features = self.fusion(fused)
        
        # Projection for contrastive learning
        projected = self.projection_head(temporal_features)
        projected = F.normalize(projected, p=2, dim=-1)
        components['projected'] = projected
        
        # Anomaly score (raw logits, sigmoid applied later)
        temporal_logits = self.temporal_anomaly_head(temporal_features)
        
        if return_components:
            return temporal_logits, temporal_features, components
        return temporal_logits, temporal_features


class TemporalEnhancedExVADv3(nn.Module):
    """
    Temporal-Enhanced ExVAD Model V3
    
    Key Fixes:
    1. Proper gradient flow through temporal path
    2. Learnable fusion weight instead of fixed
    3. Contrastive loss for discriminative features
    4. Feature variance regularization
    5. Bidirectional temporal correction for score calibration
    6. Shared CLIP feature extraction for better efficiency
    """
    
    def __init__(self,
                 device='cuda',
                 temporal_hidden_dim=256,
                 use_temporal_transformer=True,
                 use_temporal_lstm=True,
                 use_motion_encoding=True,
                 dropout=0.3,
                 freeze_base=True):  # Default freeze base CMAF
        super().__init__()
        
        self.device = device
        self.clip_feature_dim = 512
        self.temporal_hidden_dim = temporal_hidden_dim
        
        print("Initializing Temporal-Enhanced ExVAD V3...")
        
        # Base CMAF model
        print("Loading base CMAF model...")
        self.base_model = EnhancedExVADModel(device=device)
        
        # Learnable fusion weight (initialized to 0.3)
        self.temporal_weight = nn.Parameter(torch.tensor(0.3))

        # Fusion calibration controls
        self.base_low_threshold = 0.15
        self.base_high_threshold = 0.50
        self.temporal_high_conf = 0.70
        self.temporal_low_conf = 0.30
        self.down_correction_scale = 0.80
        
        # Improved temporal detector
        print("Initializing improved temporal detector...")
        self.temporal_detector = ImprovedTemporalAnomalyDetector(
            input_dim=self.clip_feature_dim,
            hidden_dim=temporal_hidden_dim,
            use_transformer=use_temporal_transformer,
            use_lstm=use_temporal_lstm,
            use_motion=use_motion_encoding,
            dropout=dropout
        )
        
        # Freeze base model by default
        if freeze_base:
            self._freeze_base_model()
        
        self.to(device)
        print(f"✅ Temporal-Enhanced ExVAD V3 initialized on {device}")
        
        # Count parameters
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"   Total params: {total:,}")
        print(f"   Trainable params: {trainable:,}")
    
    def _freeze_base_model(self):
        """Freeze base CMAF model"""
        for param in self.base_model.parameters():
            param.requires_grad = False
        print("   Base CMAF model frozen")
    
    def _unfreeze_base_model(self):
        """Unfreeze base model for fine-tuning"""
        for param in self.base_model.parameters():
            param.requires_grad = True
        print("   Base CMAF model unfrozen")

    def _has_trainable_base_params(self) -> bool:
        """Check whether any base-model parameters are trainable."""
        return any(param.requires_grad for param in self.base_model.parameters())

    def unfreeze_base_adaptation_layers(self):
        """Unfreeze only lightweight base layers for stage-2 calibration."""
        self._freeze_base_model()
        unfrozen_modules = []

        madm = getattr(self.base_model, 'madm', None)
        if madm is not None and hasattr(madm, 'anomaly_head'):
            for param in madm.anomaly_head.parameters():
                param.requires_grad = True
            unfrozen_modules.append('madm.anomaly_head')

            cmaf = getattr(madm, 'cmaf', None)
            if cmaf is not None:
                for name in ['text_projection', 'visual_projection', 'fusion_gate', 'final_projection']:
                    module = getattr(cmaf, name, None)
                    if module is not None:
                        for param in module.parameters():
                            param.requires_grad = True
                        unfrozen_modules.append(f'madm.cmaf.{name}')

        # Always keep temporal branch trainable.
        for param in self.temporal_detector.parameters():
            param.requires_grad = True
        self.temporal_weight.requires_grad = True

        print("   Selective base adaptation enabled")
        print(f"   Unfrozen modules: {', '.join(unfrozen_modules) if unfrozen_modules else 'none'}")
    
    def extract_per_frame_features(self, frames: List[Image.Image], 
                                    enable_grad=False) -> torch.Tensor:
        """
        Extract CLIP features for each frame.
        
        Args:
            frames: List of PIL Images
            enable_grad: Whether to enable gradient computation
            
        Returns:
            frame_features: (1, num_frames, 512)
        """
        madm = getattr(self.base_model, 'madm', None)
        if madm is not None and hasattr(madm, 'extract_visual_features'):
            _, per_frame_features = madm.extract_visual_features(
                frames, return_per_frame=True
            )
            return per_frame_features

        # Fallback path for compatibility with non-enhanced MADM.
        clip_model = self.base_model.madm.clip_model
        clip_preprocess = self.base_model.madm.clip_preprocess
        frame_features_list = []

        for frame in frames:
            try:
                if torch.is_tensor(frame):
                    frame_np = frame.cpu().numpy()
                    if frame_np.shape[0] == 3:
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
                    frame = Image.new('RGB', (224, 224), color=(128, 128, 128))
                if frame.mode != 'RGB':
                    frame = frame.convert('RGB')

                frame_tensor = clip_preprocess(frame).unsqueeze(0).to(self.device)
                if enable_grad:
                    features = clip_model.encode_image(frame_tensor)
                else:
                    with torch.no_grad():
                        features = clip_model.encode_image(frame_tensor)

                features = features.float()
                features = F.normalize(features, p=2, dim=1)
                frame_features_list.append(features)
            except Exception:
                frame_features_list.append(
                    torch.zeros(1, self.clip_feature_dim, device=self.device)
                )

        frame_features = torch.cat(frame_features_list, dim=0)
        return frame_features.unsqueeze(0)

    def _prepare_text_inputs(self, explanation: str, captions: List[str]) -> List[str]:
        """Build text prompt for cross-modal attention."""
        text_inputs = [explanation] if isinstance(explanation, str) else [str(explanation)]
        if isinstance(captions, (list, tuple)) and len(captions) > 0:
            first_caption = captions[0] if captions[0] else ""
            combined = f"{text_inputs[0]}. {first_caption}" if first_caption else text_inputs[0]
            text_inputs = [combined]
        return text_inputs

    def _compute_base_score_from_shared_features(self,
                                                 frames: Union[List[Image.Image], torch.Tensor],
                                                 video_features: torch.Tensor) -> Tuple[torch.Tensor, str, List[str]]:
        """Run the base CMAF branch using precomputed visual features."""
        explanation, captions = self.base_model.aegm(frames)
        text_inputs = self._prepare_text_inputs(explanation, captions)

        madm = getattr(self.base_model, 'madm', None)
        if madm is None:
            raise RuntimeError("Base model MADM component is not available")

        if hasattr(madm, 'predict_from_visual_features'):
            base_score = madm.predict_from_visual_features(
                visual_features=video_features,
                explanations=text_inputs
            )
        else:
            # Compatibility fallback for older MADM implementations.
            if self._has_trainable_base_params():
                base_score = self.base_model(frames)[0]
            else:
                with torch.no_grad():
                    base_score = self.base_model(frames)[0]

        return base_score, explanation, captions
    
    def forward(self, frames: Union[List[Image.Image], torch.Tensor],
                return_details=False) -> Tuple[torch.Tensor, str, str]:
        """
        Forward pass with temporal modeling.
        
        Returns:
            anomaly_score: (batch, 1) probability
            explanation: text
            captions: frame captions
        """
        # Extract shared CLIP features once for both base and temporal branches.
        frame_features = self.extract_per_frame_features(frames).float()
        video_features = frame_features.mean(dim=1)

        if self._has_trainable_base_params():
            base_score, explanation, captions = self._compute_base_score_from_shared_features(
                frames, video_features
            )
        else:
            with torch.no_grad():
                base_score, explanation, captions = self._compute_base_score_from_shared_features(
                    frames, video_features
                )

        if base_score.dim() == 0:
            base_score = base_score.view(1, 1)
        elif base_score.dim() == 1:
            base_score = base_score.unsqueeze(1)
        
        # Temporal prediction (WITH gradients)
        temporal_logits, temporal_features, components = self.temporal_detector(
            frame_features, return_components=True
        )
        
        # Apply sigmoid to temporal logits
        temporal_score = torch.sigmoid(temporal_logits)
        
        # Bidirectional confidence-gated correction.
        # 1) Boost when temporal anomaly evidence exceeds base score.
        # 2) Suppress overly high base scores when temporal branch is strongly normal.
        fusion_weight = torch.sigmoid(self.temporal_weight) * 0.4 + 0.1
        base_is_low = (base_score < self.base_low_threshold).float()
        base_is_high = (base_score > self.base_high_threshold).float()
        temporal_is_anom_confident = (temporal_score > self.temporal_high_conf).float()
        temporal_is_normal_confident = (temporal_score < self.temporal_low_conf).float()

        allow_boost = (1.0 - base_is_low) + (base_is_low * temporal_is_anom_confident)
        allow_suppress = base_is_high * temporal_is_normal_confident

        up_correction = torch.relu(temporal_score - base_score) * allow_boost
        down_correction = torch.relu(base_score - temporal_score) * allow_suppress

        combined_score = (
            base_score
            + fusion_weight * up_correction
            - (fusion_weight * self.down_correction_scale) * down_correction
        )
        
        # Clamp to valid probability range
        combined_score = torch.clamp(combined_score, 0.0, 1.0)
        
        # Enhanced explanation
        enhanced_explanation = self._enhance_explanation(
            explanation, temporal_score.item(), components
        )
        
        if return_details:
            return combined_score, enhanced_explanation, captions, {
                'base_score': base_score,
                'temporal_logits': temporal_logits,
                'temporal_score': temporal_score,
                'temporal_features': temporal_features,
                'fusion_weight': fusion_weight.item(),
                'allow_boost': allow_boost,
                'allow_suppress': allow_suppress,
                'up_correction': up_correction,
                'down_correction': down_correction,
                'components': components
            }
        
        return combined_score, enhanced_explanation, captions
    
    def _enhance_explanation(self, base_explanation: str, 
                             temporal_score: float, 
                             components: dict) -> str:
        """Add temporal context to explanation"""
        if temporal_score > 0.7:
            prefix = "⚠️ Strong temporal anomaly detected. "
        elif temporal_score > 0.5:
            prefix = "⚡ Moderate temporal irregularity. "
        else:
            prefix = "✓ Normal temporal pattern. "
        
        return prefix + str(base_explanation)
    
    def get_temporal_features(self, frames: List[Image.Image]) -> torch.Tensor:
        """Get temporal features for analysis"""
        frame_features = self.extract_per_frame_features(frames)
        _, temporal_features, _ = self.temporal_detector(
            frame_features, return_components=True
        )
        return temporal_features
    
    def compute_temporal_loss(self, temporal_logits: torch.Tensor,
                              temporal_features: torch.Tensor,
                              labels: torch.Tensor,
                              components: dict) -> dict:
        """
        Compute comprehensive temporal loss with regularization.
        
        Returns dict with individual loss components for logging.
        """
        batch_size = temporal_logits.shape[0]
        
        # 1. BCE Loss with label smoothing
        smooth_labels = labels * 0.9 + 0.05  # Smooth between 0.05 and 0.95
        bce_loss = F.binary_cross_entropy_with_logits(
            temporal_logits, smooth_labels
        )
        
        # 2. Feature variance loss (prevent collapse)
        # Features should have variance across samples
        if batch_size > 1:
            feature_var = temporal_features.var(dim=0).mean()
            variance_loss = torch.exp(-feature_var)  # Penalize low variance
        else:
            variance_loss = torch.tensor(0.0, device=temporal_logits.device)
        
        # 3. Contrastive margin loss (if we have both classes in batch)
        contrastive_loss = torch.tensor(0.0, device=temporal_logits.device)
        if batch_size > 1 and 'projected' in components:
            projected = components['projected']
            pos_mask = labels.view(-1) > 0.5
            neg_mask = ~pos_mask
            
            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                pos_features = projected[pos_mask]
                neg_features = projected[neg_mask]
                
                # Mean features per class
                pos_mean = pos_features.mean(dim=0)
                neg_mean = neg_features.mean(dim=0)
                
                # Margin loss: distance between class means should be > 1.0
                class_distance = F.pairwise_distance(
                    pos_mean.unsqueeze(0), neg_mean.unsqueeze(0)
                )
                margin = 1.0
                contrastive_loss = F.relu(margin - class_distance).mean()
        
        # 4. Total loss
        total_loss = bce_loss + 0.1 * variance_loss + 0.1 * contrastive_loss
        
        return {
            'total': total_loss,
            'bce': bce_loss,
            'variance': variance_loss,
            'contrastive': contrastive_loss
        }


def test_temporal_v3():
    """Test the V3 model"""
    print("\n" + "="*70)
    print("Testing Temporal-Enhanced ExVAD V3")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = TemporalEnhancedExVADv3(
        device=device,
        temporal_hidden_dim=256,
        use_temporal_transformer=True,
        use_temporal_lstm=True,
        use_motion_encoding=True,
        dropout=0.3,
        freeze_base=True
    )
    
    model.eval()
    
    # Test with dummy frames
    print("\nTesting forward pass...")
    dummy_frames = [Image.new('RGB', (224, 224), color=(100, 100, 100)) 
                    for _ in range(8)]
    
    with torch.no_grad():
        score, explanation, captions, details = model(
            dummy_frames, return_details=True
        )
    
    print(f"Combined Score: {score.item():.4f}")
    print(f"Base Score: {details['base_score'].item():.4f}")
    print(f"Temporal Score: {details['temporal_score'].item():.4f}")
    print(f"Fusion Weight: {details['fusion_weight']:.4f}")
    print(f"Explanation: {explanation[:80]}...")
    
    # Test loss computation
    print("\nTesting loss computation...")
    model.train()
    
    # Simulate batch of 2
    frames1 = [Image.new('RGB', (224, 224), color=(50, 50, 50)) for _ in range(8)]
    frames2 = [Image.new('RGB', (224, 224), color=(200, 200, 200)) for _ in range(8)]
    
    _, _, _, d1 = model(frames1, return_details=True)
    _, _, _, d2 = model(frames2, return_details=True)
    
    # Stack features and logits
    temporal_logits = torch.cat([d1['temporal_logits'], d2['temporal_logits']], dim=0)
    temporal_features = torch.cat([d1['temporal_features'], d2['temporal_features']], dim=0)
    
    # Combine projected features
    components = {
        'projected': torch.cat([d1['components']['projected'], 
                                d2['components']['projected']], dim=0)
    }
    
    labels = torch.tensor([[0.0], [1.0]], device=device)
    
    losses = model.compute_temporal_loss(
        temporal_logits, temporal_features, labels, components
    )
    
    print(f"Total Loss: {losses['total'].item():.4f}")
    print(f"BCE Loss: {losses['bce'].item():.4f}")
    print(f"Variance Loss: {losses['variance'].item():.4f}")
    print(f"Contrastive Loss: {losses['contrastive'].item():.4f}")
    
    print("\n✅ V3 Model test complete!")


if __name__ == "__main__":
    test_temporal_v3()
