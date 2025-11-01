"""
Cross-Modal Attention Fusion (CMAF) for Video Anomaly Detection
Novel contribution: Learnable attention mechanism between CLIP visual features and BLIP text embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')

class CrossModalAttentionFusion(nn.Module):
    """
    Cross-Modal Attention Fusion module that learns to attend between
    visual features (from CLIP) and textual explanations (from BLIP)
    """
    def __init__(self, visual_dim=512, text_dim=768, hidden_dim=512, num_heads=8):
        super().__init__()
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        
        # Text encoder for BLIP explanations
        self.text_encoder = AutoModel.from_pretrained("distilbert-base-uncased")
        self.text_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        # Project text features to visual space
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Project visual features to hidden space
        self.visual_projection = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Cross-attention mechanisms
        self.visual_to_text_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            batch_first=True,
            dropout=0.1
        )
        
        self.text_to_visual_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            batch_first=True,
            dropout=0.1
        )
        
        # Fusion layers
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.final_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, visual_dim)  # Back to original visual dim
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for module in [self.text_projection, self.visual_projection, self.final_projection]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def encode_text(self, explanations, device):
        """
        Encode textual explanations using DistilBERT
        Args:
            explanations: List of explanation strings
            device: Target device
        Returns:
            text_embeddings: Tensor of shape [batch_size, text_dim]
        """
        if not explanations or all(exp is None for exp in explanations):
            # Return zero embeddings if no explanations
            return torch.zeros(len(explanations) if explanations else 1, self.text_dim).to(device)
        
        # Handle None explanations
        processed_explanations = [exp if exp is not None else "no explanation" for exp in explanations]
        
        # Tokenize explanations
        tokens = self.text_tokenizer(
            processed_explanations,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Move to device
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        # Get text embeddings
        with torch.no_grad():
            text_outputs = self.text_encoder(**tokens)
            # Use [CLS] token embedding
            text_embeddings = text_outputs.last_hidden_state[:, 0, :]  # [batch_size, text_dim]
        
        return text_embeddings
    
    def forward(self, visual_features, explanations, device):
        """
        Forward pass through cross-modal attention fusion
        Args:
            visual_features: Tensor of shape [batch_size, visual_dim]
            explanations: List of explanation strings
            device: Target device
        Returns:
            fused_features: Enhanced visual features [batch_size, visual_dim]
        """
        batch_size = visual_features.size(0)
        
        # Ensure visual features are in float32 for compatibility
        visual_features = visual_features.float()
        
        # Encode text explanations
        text_embeddings = self.encode_text(explanations, device)  # [batch_size, text_dim]
        text_embeddings = text_embeddings.float()  # Ensure float32
        
        # Project to hidden space
        visual_proj = self.visual_projection(visual_features)  # [batch_size, hidden_dim]
        text_proj = self.text_projection(text_embeddings)      # [batch_size, hidden_dim]
        
        # Add sequence dimension for attention (treating each sample as a sequence of length 1)
        visual_seq = visual_proj.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        text_seq = text_proj.unsqueeze(1)      # [batch_size, 1, hidden_dim]
        
        # Cross-attention: Visual attending to Text
        visual_attended, _ = self.visual_to_text_attention(
            query=visual_seq,      # Visual features as queries
            key=text_seq,          # Text features as keys
            value=text_seq         # Text features as values
        )
        visual_attended = visual_attended.squeeze(1)  # [batch_size, hidden_dim]
        
        # Cross-attention: Text attending to Visual
        text_attended, _ = self.text_to_visual_attention(
            query=text_seq,        # Text features as queries
            key=visual_seq,        # Visual features as keys
            value=visual_seq       # Visual features as values
        )
        text_attended = text_attended.squeeze(1)  # [batch_size, hidden_dim]
        
        # Fusion with gating mechanism
        combined_features = torch.cat([visual_attended, text_attended], dim=-1)  # [batch_size, hidden_dim*2]
        gate = self.fusion_gate(combined_features)  # [batch_size, hidden_dim]
        
        # Gated fusion
        fused_hidden = gate * visual_attended + (1 - gate) * text_attended  # [batch_size, hidden_dim]
        
        # Final projection back to visual space
        fused_features = self.final_projection(fused_hidden)  # [batch_size, visual_dim]
        
        # Residual connection - ensure dtype compatibility
        enhanced_features = visual_features + fused_features
        
        # Convert back to original dtype if needed
        if hasattr(visual_features, 'dtype') and visual_features.dtype == torch.float16:
            enhanced_features = enhanced_features.half()
        
        return enhanced_features

class EnhancedMADM(nn.Module):
    """
    Enhanced MADM with Cross-Modal Attention Fusion
    """
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        # Original CLIP components
        import clip
        print(f"Loading CLIP model on {device}...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        self.clip_model.eval()
        
        # Cross-modal attention fusion
        self.cmaf = CrossModalAttentionFusion(
            visual_dim=512,
            text_dim=768,
            hidden_dim=512,
            num_heads=8
        ).to(device)
        
        # Enhanced anomaly detection head
        self.anomaly_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        ).to(device)
        
        print("✅ Enhanced MADM with Cross-Modal Attention loaded!")
    
    def extract_visual_features(self, frames):
        """Extract CLIP features from frames (same as original)"""
        features_list = []
        
        with torch.no_grad():
            for frame in frames:
                try:
                    # Ensure frame is PIL Image
                    if not hasattr(frame, 'mode'):
                        if torch.is_tensor(frame):
                            frame_array = frame.cpu().numpy()
                        else:
                            frame_array = frame
                        
                        import numpy as np
                        from PIL import Image
                        if isinstance(frame_array, np.ndarray):
                            if frame_array.dtype != np.uint8:
                                frame_array = (frame_array * 255).astype(np.uint8)
                            frame = Image.fromarray(frame_array)
                        else:
                            frame = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
                    
                    if frame.mode != 'RGB':
                        frame = frame.convert('RGB')
                    
                    # Preprocess and extract features
                    frame_tensor = self.clip_preprocess(frame).unsqueeze(0).to(self.device)
                    features = self.clip_model.encode_image(frame_tensor)
                    features = F.normalize(features, p=2, dim=1)
                    features_list.append(features)
                    
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    dummy_features = torch.zeros(1, 512).to(self.device)
                    features_list.append(dummy_features)
        
        if features_list:
            video_features = torch.stack(features_list, dim=1).mean(dim=1)
        else:
            video_features = torch.zeros(1, 512).to(self.device)
        
        return video_features
    
    def forward(self, frames, explanations=None):
        """
        Enhanced forward pass with cross-modal attention
        Args:
            frames: List of video frames
            explanations: List of textual explanations from BLIP
        Returns:
            anomaly_score: Predicted anomaly score
        """
        # Extract visual features
        visual_features = self.extract_visual_features(frames)
        original_dtype = visual_features.dtype
        
        # Apply cross-modal attention fusion if explanations are provided
        if explanations is not None:
            enhanced_features = self.cmaf(visual_features, explanations, self.device)
        else:
            enhanced_features = visual_features
        
        # Ensure consistent dtype with anomaly head parameters
        target_dtype = next(self.anomaly_head.parameters()).dtype
        enhanced_features = enhanced_features.to(dtype=target_dtype)
        
        # Predict anomaly score
        anomaly_score = self.anomaly_head(enhanced_features)
        
        return anomaly_score
