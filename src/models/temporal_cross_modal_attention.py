"""
Temporal Cross-Modal Attention (TCMA) Implementation
Novel extension of CMAF for temporal video sequences
Target: IEEE TPAMI / CVPR publication
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')

class TemporalCrossModalAttention(nn.Module):
    """
    Novel: Temporal Cross-Modal Attention for Video Sequences
    
    Key Innovations:
    1. Cross-modal attention across temporal sequences
    2. Temporal consistency in explanations  
    3. Dynamic fusion weights over time
    4. Bidirectional temporal understanding
    """
    def __init__(self, visual_dim=512, text_dim=768, hidden_dim=512, 
                 num_heads=8, temporal_len=16, num_layers=2):
        super().__init__()
        
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.temporal_len = temporal_len
        
        # Text encoder for temporal explanations
        self.text_encoder = AutoModel.from_pretrained("distilbert-base-uncased")
        self.text_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        # Temporal encoding layers
        self.temporal_visual_encoder = nn.LSTM(
            visual_dim, hidden_dim//2, num_layers=num_layers, 
            bidirectional=True, batch_first=True, dropout=0.1
        )
        
        self.temporal_text_encoder = nn.LSTM(
            text_dim, hidden_dim//2, num_layers=num_layers,
            bidirectional=True, batch_first=True, dropout=0.1
        )
        
        # Temporal cross-attention mechanisms
        self.visual_to_text_temporal = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, 
            batch_first=True, dropout=0.1
        )
        
        self.text_to_visual_temporal = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            batch_first=True, dropout=0.1
        )
        
        # Temporal fusion with adaptive gates
        self.temporal_fusion_gate = nn.Sequential(
            nn.LSTM(hidden_dim*2, hidden_dim, batch_first=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Temporal consistency regularization
        self.temporal_consistency = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )
        
        # Final temporal projection
        self.temporal_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, visual_dim)
        )
        
        # Positional encoding for temporal sequences
        self.register_buffer(
            "temporal_pos_encoding", 
            self._generate_temporal_encoding(temporal_len, hidden_dim)
        )
        
        self._init_weights()
    
    def _generate_temporal_encoding(self, max_len, d_model):
        """Generate sinusoidal positional encoding for temporal sequences"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # [1, max_len, d_model]
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for module in [self.temporal_fusion_gate, self.temporal_consistency, 
                      self.temporal_projection]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def encode_temporal_text(self, explanation_sequences, device):
        """
        Encode temporal sequence of explanations
        Args:
            explanation_sequences: List of lists [[exp1_t1, exp1_t2, ...], [exp2_t1, exp2_t2, ...]]
            device: Target device
        Returns:
            temporal_text_embeddings: [batch_size, temporal_len, text_dim]
        """
        batch_size = len(explanation_sequences)
        temporal_embeddings = []
        
        for sequence in explanation_sequences:
            sequence_embeddings = []
            
            for explanation in sequence:
                if explanation is None:
                    explanation = "no explanation available"
                
                # Tokenize and encode
                tokens = self.text_tokenizer(
                    explanation, padding=True, truncation=True,
                    max_length=64, return_tensors="pt"
                )
                tokens = {k: v.to(device) for k, v in tokens.items()}
                
                with torch.no_grad():
                    text_output = self.text_encoder(**tokens)
                    embedding = text_output.last_hidden_state[:, 0, :]  # [CLS] token
                    sequence_embeddings.append(embedding)
            
            # Stack temporal sequence
            temporal_seq = torch.cat(sequence_embeddings, dim=0)  # [temporal_len, text_dim]
            temporal_embeddings.append(temporal_seq)
        
        # Stack batch
        temporal_text = torch.stack(temporal_embeddings, dim=0)  # [batch, temporal_len, text_dim]
        
        return temporal_text.float()
    
    def forward(self, visual_sequence, explanation_sequences, device):
        """
        Forward pass through temporal cross-modal attention
        Args:
            visual_sequence: [batch_size, temporal_len, visual_dim] - CLIP features over time
            explanation_sequences: List of temporal explanation sequences
            device: Target device
        Returns:
            enhanced_temporal_features: [batch_size, temporal_len, visual_dim]
            attention_weights: Temporal attention patterns
        """
        batch_size, temporal_len, _ = visual_sequence.shape
        
        # Ensure visual features are float32
        visual_sequence = visual_sequence.float()
        
        # Encode temporal text explanations
        text_sequence = self.encode_temporal_text(explanation_sequences, device)
        
        # Add positional encoding
        pos_encoding = self.temporal_pos_encoding[:, :temporal_len, :].to(device)
        
        # Temporal encoding through LSTM
        temporal_visual, (h_v, c_v) = self.temporal_visual_encoder(visual_sequence)
        temporal_text, (h_t, c_t) = self.temporal_text_encoder(text_sequence)
        
        # Add positional encoding
        temporal_visual = temporal_visual + pos_encoding
        temporal_text = temporal_text + pos_encoding
        
        # Bidirectional temporal cross-attention
        # Visual attending to text across time
        visual_attended, visual_attention_weights = self.visual_to_text_temporal(
            query=temporal_visual,      # How visual evolves over time
            key=temporal_text,          # What text explains across time
            value=temporal_text         # Textual information over time
        )
        
        # Text attending to visual across time  
        text_attended, text_attention_weights = self.text_to_visual_temporal(
            query=temporal_text,        # How text evolves over time
            key=temporal_visual,        # What visual shows across time
            value=temporal_visual       # Visual information over time
        )
        
        # Temporal fusion with adaptive gates
        combined_temporal = torch.cat([visual_attended, text_attended], dim=-1)
        
        # Learn temporal fusion weights
        fusion_output, _ = self.temporal_fusion_gate[0](combined_temporal)  # LSTM
        temporal_gates = self.temporal_fusion_gate[1:](fusion_output)  # Linear + Sigmoid
        
        # Gated temporal fusion
        fused_temporal = (temporal_gates * visual_attended + 
                         (1 - temporal_gates) * text_attended)
        
        # Temporal consistency regularization
        consistency_scores = self.temporal_consistency(fused_temporal)
        
        # Final projection back to visual space
        enhanced_features = self.temporal_projection(fused_temporal)
        
        # Residual connection with original visual sequence
        final_features = visual_sequence + enhanced_features
        
        return final_features, {
            'visual_attention': visual_attention_weights,
            'text_attention': text_attention_weights,
            'temporal_gates': temporal_gates,
            'consistency_scores': consistency_scores
        }

class EnhancedTemporalMADM(nn.Module):
    """
    Enhanced MADM with Temporal Cross-Modal Attention
    """
    def __init__(self, device='cuda', temporal_len=16):
        super().__init__()
        self.device = device
        self.temporal_len = temporal_len
        
        # Original CLIP components
        import clip
        print(f"Loading CLIP model on {device}...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        self.clip_model.eval()
        
        # Temporal cross-modal attention
        self.tcma = TemporalCrossModalAttention(
            visual_dim=512,
            text_dim=768,
            hidden_dim=512,
            num_heads=8,
            temporal_len=temporal_len
        ).to(device)
        
        # Temporal anomaly detection head
        self.temporal_anomaly_head = nn.Sequential(
            nn.LSTM(512, 256, num_layers=2, batch_first=True, dropout=0.1),
            nn.Flatten(),
            nn.Linear(256 * temporal_len, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        ).to(device)
        
        print("✅ Enhanced Temporal MADM with TCMA loaded!")
    
    def extract_temporal_visual_features(self, frame_sequences):
        """
        Extract CLIP features from temporal frame sequences
        Args:
            frame_sequences: List of frame sequences [[frames_video1], [frames_video2], ...]
        Returns:
            temporal_features: [batch_size, temporal_len, 512]
        """
        batch_features = []
        
        with torch.no_grad():
            for frame_sequence in frame_sequences:
                sequence_features = []
                
                for frame in frame_sequence:
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
                        
                        if frame.mode != 'RGB':
                            frame = frame.convert('RGB')
                        
                        # Extract CLIP features
                        frame_tensor = self.clip_preprocess(frame).unsqueeze(0).to(self.device)
                        features = self.clip_model.encode_image(frame_tensor)
                        features = F.normalize(features, p=2, dim=1)
                        sequence_features.append(features)
                        
                    except Exception as e:
                        print(f"Error processing frame: {e}")
                        dummy_features = torch.zeros(1, 512).to(self.device)
                        sequence_features.append(dummy_features)
                
                # Stack temporal sequence
                if sequence_features:
                    temporal_seq = torch.stack(sequence_features, dim=1).squeeze(0)  # [temporal_len, 512]
                else:
                    temporal_seq = torch.zeros(self.temporal_len, 512).to(self.device)
                
                batch_features.append(temporal_seq)
        
        # Stack batch
        temporal_features = torch.stack(batch_features, dim=0)  # [batch, temporal_len, 512]
        
        return temporal_features
    
    def forward(self, frame_sequences, explanation_sequences):
        """
        Enhanced forward pass with temporal cross-modal attention
        Args:
            frame_sequences: List of temporal frame sequences
            explanation_sequences: List of temporal explanation sequences
        Returns:
            anomaly_score: Predicted anomaly score considering temporal dynamics
        """
        # Extract temporal visual features
        temporal_visual = self.extract_temporal_visual_features(frame_sequences)
        
        # Apply temporal cross-modal attention
        enhanced_temporal, attention_info = self.tcma(
            temporal_visual, explanation_sequences, self.device
        )
        
        # Temporal anomaly detection
        anomaly_score = self.temporal_anomaly_head(enhanced_temporal)
        
        return anomaly_score, attention_info

def temporal_cross_modal_loss(temporal_features, consistency_scores, alpha=0.1):
    """
    Novel loss function for temporal cross-modal consistency
    Args:
        temporal_features: [batch, temporal_len, dim]
        consistency_scores: [batch, temporal_len, 1]
        alpha: Consistency regularization weight
    """
    # Temporal smoothness loss
    temporal_diff = temporal_features[:, 1:] - temporal_features[:, :-1]
    smoothness_loss = torch.mean(torch.norm(temporal_diff, dim=-1))
    
    # Consistency regularization
    consistency_loss = torch.mean(torch.abs(consistency_scores))
    
    return alpha * (smoothness_loss + consistency_loss)

# Example usage for training
def train_temporal_cmaf_epoch(model, dataloader, optimizer, device):
    """Training loop with temporal cross-modal attention"""
    model.train()
    total_loss = 0
    
    for batch_idx, (frame_sequences, explanation_sequences, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Forward pass
        anomaly_scores, attention_info = model(frame_sequences, explanation_sequences)
        
        # Main anomaly detection loss
        main_loss = F.binary_cross_entropy_with_logits(
            anomaly_scores.squeeze(), labels.float().to(device)
        )
        
        # Temporal consistency loss
        temporal_loss = temporal_cross_modal_loss(
            attention_info['visual_attention'], 
            attention_info['consistency_scores']
        )
        
        # Total loss
        total_loss_batch = main_loss + temporal_loss
        total_loss_batch.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += total_loss_batch.item()
        
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}, Loss: {total_loss_batch.item():.4f}')
    
    return total_loss / len(dataloader)

if __name__ == "__main__":
    # Example initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create enhanced temporal model
    temporal_model = EnhancedTemporalMADM(device=device, temporal_len=16)
    
    print("✅ Temporal Cross-Modal Attention model initialized!")
    print("🎯 Novel contribution ready for IEEE TPAMI/CVPR submission!")
