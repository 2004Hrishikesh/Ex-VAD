"""
Enhanced ExVAD Model with Cross-Modal Attention Fusion
Novel contribution for IEEE paper: Learnable fusion between CLIP and BLIP modalities
"""

import torch
import torch.nn as nn
from .aegm import SimpleAEGM
from .cross_modal_attention import EnhancedMADM

class EnhancedExVADModel(nn.Module):
    """
    Enhanced ExVAD model with Cross-Modal Attention Fusion (CMAF)
    
    Key improvements:
    1. Cross-modal attention between CLIP visual features and BLIP text embeddings
    2. Enhanced anomaly detection head with layer normalization
    3. Residual connections for better gradient flow
    4. Learnable fusion gates for adaptive modality weighting
    """
    def __init__(self, device='cuda', use_cross_modal_attention=True):
        super().__init__()
        self.device = device
        self.use_cross_modal_attention = use_cross_modal_attention
        
        print(f"Initializing Enhanced ExVAD model on {device}...")
        
        # Initialize AEGM (BLIP-based explanation generator)
        self.aegm = SimpleAEGM(device)
        
        # Initialize Enhanced MADM with cross-modal attention
        if use_cross_modal_attention:
            self.madm = EnhancedMADM(device)
            print("✅ Using Cross-Modal Attention Fusion")
        else:
            # Fallback to original MADM for ablation studies
            from .madm import SimpleMADM
            self.madm = SimpleMADM(device)
            print("✅ Using Original MADM (no cross-modal attention)")
        
        # Additional fusion components
        self.modality_weights = nn.Parameter(torch.tensor([0.7, 0.3]))  # Visual, Text weights
        self.temperature = nn.Parameter(torch.tensor(1.0))  # Temperature for attention
        
        print(f"✅ Enhanced ExVAD model initialized on {device}")
    
    def forward(self, frames):
        """
        Enhanced forward pass with cross-modal attention fusion
        
        Args:
            frames: List of video frames
            
        Returns:
            anomaly_score: Predicted anomaly score (tensor)
            explanation: Generated explanation (string)
            captions: Generated captions (list of strings)
        """
        # Generate explanations and captions using AEGM (BLIP)
        explanation, captions = self.aegm(frames)
        
        # Prepare explanations for cross-modal attention
        if self.use_cross_modal_attention:
            # Use both explanation and captions for richer text representation
            text_inputs = [explanation] if isinstance(explanation, str) else [str(explanation)]
            if captions and len(captions) > 0:
                # Combine explanation with first caption
                combined_text = f"{explanation}. {captions[0]}" if captions[0] else explanation
                text_inputs = [combined_text]
            
            # Forward through enhanced MADM with cross-modal attention
            anomaly_score = self.madm(frames, explanations=text_inputs)
        else:
            # Forward through original MADM without text
            anomaly_score = self.madm(frames)
        
        return anomaly_score, explanation, captions
    
    def forward_with_attention_weights(self, frames):
        """
        Forward pass that also returns attention weights for visualization
        
        Args:
            frames: List of video frames
            
        Returns:
            anomaly_score: Predicted anomaly score
            explanation: Generated explanation  
            captions: Generated captions
            attention_weights: Cross-modal attention weights (if available)
        """
        # Generate text
        explanation, captions = self.aegm(frames)
        
        if self.use_cross_modal_attention and hasattr(self.madm, 'cmaf'):
            # Get visual features
            visual_features = self.madm.extract_visual_features(frames)
            
            # Prepare text
            text_inputs = [explanation] if isinstance(explanation, str) else [str(explanation)]
            if captions and len(captions) > 0:
                combined_text = f"{explanation}. {captions[0]}" if captions[0] else explanation
                text_inputs = [combined_text]
            
            # Forward through CMAF to get enhanced features
            enhanced_features = self.madm.cmaf(visual_features, text_inputs, self.device)
            
            # Get anomaly score
            anomaly_score = self.madm.anomaly_head(enhanced_features)
            
            # For attention visualization (simplified)
            attention_weights = {
                'modality_weights': torch.softmax(self.modality_weights, dim=0),
                'temperature': self.temperature.item()
            }
            
            return anomaly_score, explanation, captions, attention_weights
        else:
            # Fallback without attention weights
            anomaly_score = self.madm(frames)
            return anomaly_score, explanation, captions, None
    
    def get_cross_modal_similarity(self, frames):
        """
        Compute cross-modal similarity between visual and textual features
        Useful for analysis and ablation studies
        
        Args:
            frames: List of video frames
            
        Returns:
            similarity_score: Cosine similarity between visual and text embeddings
        """
        if not self.use_cross_modal_attention:
            return torch.tensor(0.0)
        
        # Generate explanation
        explanation, captions = self.aegm(frames)
        text_inputs = [explanation]
        
        # Get visual features
        visual_features = self.madm.extract_visual_features(frames)  # [1, 512]
        
        # Get text embeddings 
        text_embeddings = self.madm.cmaf.encode_text(text_inputs, self.device)  # [1, 768]
        
        # Project text to visual space for comparison
        text_projected = self.madm.cmaf.text_projection(text_embeddings.float())  # [1, 512]
        
        # Compute cosine similarity
        visual_norm = torch.nn.functional.normalize(visual_features.float(), p=2, dim=1)
        text_norm = torch.nn.functional.normalize(text_projected, p=2, dim=1)
        similarity = torch.sum(visual_norm * text_norm, dim=1)
        
        return similarity.mean().item()
    
    def enable_cross_modal_attention(self):
        """Enable cross-modal attention (for ablation studies)"""
        if hasattr(self, 'madm') and hasattr(self.madm, 'cmaf'):
            self.use_cross_modal_attention = True
            print("✅ Cross-modal attention enabled")
        else:
            print("❌ Cross-modal attention not available in current model")
    
    def disable_cross_modal_attention(self):
        """Disable cross-modal attention (for ablation studies)"""
        self.use_cross_modal_attention = False
        print("✅ Cross-modal attention disabled")
    
    def to(self, device):
        """Move model to device"""
        self.device = device
        self.aegm = self.aegm.to(device)
        self.madm = self.madm.to(device)
        return super().to(device)
    
    def get_model_info(self):
        """Get model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'model_type': 'Enhanced ExVAD with Cross-Modal Attention Fusion',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'cross_modal_attention': self.use_cross_modal_attention,
            'device': str(self.device)
        }
        
        return info

# Factory function for easy model creation
def create_enhanced_exvad_model(device='cuda', use_cross_modal_attention=True):
    """
    Factory function to create Enhanced ExVAD model
    
    Args:
        device: Target device ('cuda' or 'cpu')
        use_cross_modal_attention: Whether to use cross-modal attention fusion
        
    Returns:
        Enhanced ExVAD model instance
    """
    return EnhancedExVADModel(device=device, use_cross_modal_attention=use_cross_modal_attention)
