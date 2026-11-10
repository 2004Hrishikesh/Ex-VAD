"""
Temporal Modeling Module for Ex-VAD
Implements various temporal modeling approaches to capture frame-to-frame dynamics
and sequential patterns in video anomaly detection.

Key Components:
1. TemporalTransformer - Multi-head self-attention for temporal relationships
2. TemporalLSTM - Bidirectional LSTM for sequential modeling
3. TemporalConv1D - 1D convolutions for local temporal patterns
4. MotionEncoder - Lightweight motion/flow estimation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences"""
    
    def __init__(self, d_model, max_len=64, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    """
    Transformer-based temporal modeling for video frames.
    Uses multi-head self-attention to capture long-range temporal dependencies.
    """
    
    def __init__(self, 
                 input_dim=768,
                 hidden_dim=512,
                 num_heads=8,
                 num_layers=3,
                 dropout=0.1,
                 max_frames=32):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Project input to hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=max_frames, dropout=dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Temporal attention pooling
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, frame_features, return_attention=False):
        """
        Args:
            frame_features: (batch, num_frames, input_dim) - per-frame features
            return_attention: Whether to return attention weights
            
        Returns:
            temporal_features: (batch, hidden_dim) - aggregated temporal features
            frame_outputs: (batch, num_frames, hidden_dim) - per-frame outputs
            attention_weights: (batch, num_frames) - temporal attention weights (optional)
        """
        batch_size, num_frames, _ = frame_features.shape
        
        # Project input
        x = self.input_proj(frame_features)  # (batch, frames, hidden_dim)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        frame_outputs = self.transformer(x)  # (batch, frames, hidden_dim)
        frame_outputs = self.layer_norm(frame_outputs)
        
        # Temporal attention pooling
        attn_scores = self.temporal_attention(frame_outputs).squeeze(-1)  # (batch, frames)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, frames)
        
        # Weighted aggregation
        temporal_features = torch.bmm(
            attn_weights.unsqueeze(1),  # (batch, 1, frames)
            frame_outputs               # (batch, frames, hidden_dim)
        ).squeeze(1)  # (batch, hidden_dim)
        
        temporal_features = self.output_proj(temporal_features)
        
        if return_attention:
            return temporal_features, frame_outputs, attn_weights
        return temporal_features, frame_outputs


class TemporalLSTM(nn.Module):
    """
    Bidirectional LSTM for temporal modeling.
    Captures both forward and backward temporal context.
    """
    
    def __init__(self,
                 input_dim=768,
                 hidden_dim=256,
                 num_layers=2,
                 dropout=0.2,
                 bidirectional=True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output dimension
        self.output_dim = hidden_dim * self.num_directions
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Attention for aggregation
        self.attention = nn.Sequential(
            nn.Linear(self.output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, frame_features, return_hidden=False):
        """
        Args:
            frame_features: (batch, num_frames, input_dim)
            return_hidden: Whether to return LSTM hidden states
            
        Returns:
            temporal_features: (batch, hidden_dim) - aggregated features
            frame_outputs: (batch, num_frames, output_dim) - per-frame outputs
        """
        batch_size = frame_features.size(0)
        
        # Project input
        x = self.input_proj(frame_features)  # (batch, frames, hidden_dim)
        
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch, frames, hidden_dim * directions)
        
        # Attention pooling
        attn_scores = self.attention(lstm_out).squeeze(-1)  # (batch, frames)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Weighted aggregation
        temporal_features = torch.bmm(
            attn_weights.unsqueeze(1),
            lstm_out
        ).squeeze(1)  # (batch, output_dim)
        
        temporal_features = self.output_proj(temporal_features)  # (batch, hidden_dim)
        
        if return_hidden:
            return temporal_features, lstm_out, (h_n, c_n)
        return temporal_features, lstm_out


class TemporalConv1D(nn.Module):
    """
    1D Convolutional temporal modeling for local temporal patterns.
    Uses dilated convolutions to capture multi-scale temporal information.
    """
    
    def __init__(self,
                 input_dim=768,
                 hidden_dim=256,
                 num_layers=4,
                 kernel_size=3,
                 dropout=0.2):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Dilated temporal convolutions
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size, 
                              padding=padding, dilation=dilation),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
            )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, frame_features):
        """
        Args:
            frame_features: (batch, num_frames, input_dim)
            
        Returns:
            temporal_features: (batch, hidden_dim)
            frame_outputs: (batch, num_frames, hidden_dim)
        """
        # Project and transpose for conv1d
        x = self.input_proj(frame_features)  # (batch, frames, hidden_dim)
        x = x.transpose(1, 2)  # (batch, hidden_dim, frames)
        
        # Apply dilated convolutions with residual connections
        for conv in self.conv_layers:
            residual = x
            x = conv(x) + residual
        
        x = x.transpose(1, 2)  # (batch, frames, hidden_dim)
        
        # Global average pooling
        temporal_features = x.mean(dim=1)  # (batch, hidden_dim)
        temporal_features = self.output_proj(temporal_features)
        
        return temporal_features, x


class MotionEncoder(nn.Module):
    """
    Lightweight motion encoding from frame differences.
    Captures motion patterns without explicit optical flow computation.
    """
    
    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Frame difference encoder - use input_dim instead of hardcoded 768
        self.diff_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Motion aggregation
        self.motion_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, frame_features):
        """
        Args:
            frame_features: (batch, num_frames, feature_dim) - CLIP/visual features per frame
            
        Returns:
            motion_features: (batch, hidden_dim)
        """
        batch_size, num_frames, feat_dim = frame_features.shape
        
        # Compute frame differences (motion approximation)
        frame_diffs = frame_features[:, 1:, :] - frame_features[:, :-1, :]  # (batch, frames-1, feat_dim)
        
        # Encode differences
        motion_encoded = self.diff_encoder(frame_diffs)  # (batch, frames-1, hidden_dim)
        
        # LSTM for motion sequence
        motion_out, _ = self.motion_lstm(motion_encoded)  # (batch, frames-1, hidden_dim)
        
        # Global pooling
        motion_features = motion_out.mean(dim=1)  # (batch, hidden_dim)
        motion_features = self.output_proj(motion_features)
        
        return motion_features


class TemporalAnomalyDetector(nn.Module):
    """
    Complete temporal anomaly detection module.
    Combines multiple temporal modeling approaches for robust detection.
    """
    
    def __init__(self,
                 input_dim=768,
                 hidden_dim=512,
                 use_transformer=True,
                 use_lstm=True,
                 use_motion=True,
                 num_transformer_layers=3,
                 num_lstm_layers=2,
                 dropout=0.2):
        super().__init__()
        
        self.use_transformer = use_transformer
        self.use_lstm = use_lstm
        self.use_motion = use_motion
        
        # Temporal modules
        if use_transformer:
            self.temporal_transformer = TemporalTransformer(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_transformer_layers,
                dropout=dropout
            )
        
        if use_lstm:
            self.temporal_lstm = TemporalLSTM(
                input_dim=input_dim,
                hidden_dim=hidden_dim // 2,
                num_layers=num_lstm_layers,
                dropout=dropout
            )
        
        if use_motion:
            self.motion_encoder = MotionEncoder(input_dim=input_dim, hidden_dim=hidden_dim // 2)
        
        # Calculate fusion input dimension
        fusion_dim = 0
        if use_transformer:
            fusion_dim += hidden_dim
        if use_lstm:
            fusion_dim += hidden_dim // 2
        if use_motion:
            fusion_dim += hidden_dim // 2
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Temporal anomaly score prediction
        self.temporal_anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, frame_features, return_components=False):
        """
        Args:
            frame_features: (batch, num_frames, input_dim)
            return_components: Whether to return individual component outputs
            
        Returns:
            temporal_score: (batch, 1) - temporal anomaly score
            temporal_features: (batch, hidden_dim) - fused temporal features
        """
        components = {}
        features_list = []
        
        if self.use_transformer:
            trans_feat, trans_frames = self.temporal_transformer(frame_features)
            features_list.append(trans_feat)
            components['transformer'] = trans_feat
        
        if self.use_lstm:
            lstm_feat, lstm_frames = self.temporal_lstm(frame_features)
            features_list.append(lstm_feat)
            components['lstm'] = lstm_feat
        
        if self.use_motion:
            motion_feat = self.motion_encoder(frame_features)
            features_list.append(motion_feat)
            components['motion'] = motion_feat
        
        # Fuse all temporal features
        fused = torch.cat(features_list, dim=-1)
        temporal_features = self.fusion(fused)
        
        # Predict temporal anomaly score
        temporal_score = self.temporal_anomaly_head(temporal_features)
        
        if return_components:
            return temporal_score, temporal_features, components
        return temporal_score, temporal_features


# Utility functions for integration with existing model

def extract_frame_features(clip_model, frames_tensor):
    """
    Extract per-frame CLIP features for temporal modeling.
    
    Args:
        clip_model: CLIP visual encoder
        frames_tensor: (batch, num_frames, C, H, W)
        
    Returns:
        frame_features: (batch, num_frames, feature_dim)
    """
    batch_size, num_frames, C, H, W = frames_tensor.shape
    
    # Reshape for batch processing
    frames_flat = frames_tensor.view(batch_size * num_frames, C, H, W)
    
    # Extract features
    with torch.no_grad():
        features = clip_model.encode_image(frames_flat)
    
    # Reshape back
    feature_dim = features.shape[-1]
    frame_features = features.view(batch_size, num_frames, feature_dim)
    
    return frame_features


if __name__ == "__main__":
    # Test the modules
    print("Testing Temporal Modeling Modules...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create dummy input (batch=2, frames=16, features=768)
    batch_size = 2
    num_frames = 16
    feature_dim = 768
    
    dummy_features = torch.randn(batch_size, num_frames, feature_dim).to(device)
    
    # Test TemporalTransformer
    print("\n1. Testing TemporalTransformer...")
    transformer = TemporalTransformer(input_dim=feature_dim, hidden_dim=512).to(device)
    trans_out, trans_frames, attn = transformer(dummy_features, return_attention=True)
    print(f"   Input: {dummy_features.shape}")
    print(f"   Output: {trans_out.shape}")
    print(f"   Frame outputs: {trans_frames.shape}")
    print(f"   Attention: {attn.shape}")
    
    # Test TemporalLSTM
    print("\n2. Testing TemporalLSTM...")
    lstm = TemporalLSTM(input_dim=feature_dim, hidden_dim=256).to(device)
    lstm_out, lstm_frames = lstm(dummy_features)
    print(f"   Input: {dummy_features.shape}")
    print(f"   Output: {lstm_out.shape}")
    print(f"   Frame outputs: {lstm_frames.shape}")
    
    # Test TemporalConv1D
    print("\n3. Testing TemporalConv1D...")
    conv = TemporalConv1D(input_dim=feature_dim, hidden_dim=256).to(device)
    conv_out, conv_frames = conv(dummy_features)
    print(f"   Input: {dummy_features.shape}")
    print(f"   Output: {conv_out.shape}")
    print(f"   Frame outputs: {conv_frames.shape}")
    
    # Test MotionEncoder
    print("\n4. Testing MotionEncoder...")
    motion = MotionEncoder(hidden_dim=256).to(device)
    motion_out = motion(dummy_features)
    print(f"   Input: {dummy_features.shape}")
    print(f"   Output: {motion_out.shape}")
    
    # Test Complete TemporalAnomalyDetector
    print("\n5. Testing TemporalAnomalyDetector (Full)...")
    detector = TemporalAnomalyDetector(
        input_dim=feature_dim,
        hidden_dim=512,
        use_transformer=True,
        use_lstm=True,
        use_motion=True
    ).to(device)
    
    score, features, components = detector(dummy_features, return_components=True)
    print(f"   Input: {dummy_features.shape}")
    print(f"   Temporal Score: {score.shape}")
    print(f"   Temporal Features: {features.shape}")
    print(f"   Components: {list(components.keys())}")
    
    # Parameter count
    total_params = sum(p.numel() for p in detector.parameters())
    trainable_params = sum(p.numel() for p in detector.parameters() if p.requires_grad)
    print(f"\n   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    print("\n✅ All temporal modeling modules tested successfully!")
