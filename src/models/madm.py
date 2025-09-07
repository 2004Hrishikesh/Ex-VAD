
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import numpy as np
from PIL import Image

class SimpleMADM(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        print(f"Loading CLIP model on {device}...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        self.clip_model.eval()
        
        # Anomaly detection head
        self.anomaly_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        ).to(device)
        
        print("✅ CLIP model loaded successfully!")
    
    def extract_visual_features(self, frames):
        """Extract CLIP features from frames with proper error handling"""
        features_list = []
        
        with torch.no_grad():
            for frame in frames:
                try:
                    # Ensure frame is PIL Image and handle numpy array conversion
                    if not isinstance(frame, Image.Image):
                        if torch.is_tensor(frame):
                            # Convert tensor to numpy array
                            frame_array = frame.cpu().numpy()
                        else:
                            # Direct conversion to numpy array
                            frame_array = frame

                        # Convert to PIL Image
                        if isinstance(frame_array, np.ndarray):
                            if frame_array.dtype != np.uint8:
                                frame_array = (frame_array * 255).astype(np.uint8)
                            frame = Image.fromarray(frame_array)
                        else:
                            frame = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

                    # Ensure the image has the right mode
                    if frame.mode != 'RGB':
                        frame = frame.convert('RGB')
                    
                    # Preprocess frame with CLIP
                    frame_tensor = self.clip_preprocess(frame).unsqueeze(0).to(self.device)
                    
                    # Extract features
                    features = self.clip_model.encode_image(frame_tensor)
                    features = F.normalize(features, p=2, dim=1)
                    features_list.append(features)
                    
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    # Create dummy features if processing fails
                    dummy_features = torch.zeros(1, 512).to(self.device)
                    features_list.append(dummy_features)
        
        if features_list:
            video_features = torch.stack(features_list, dim=1).mean(dim=1)
        else:
            video_features = torch.zeros(1, 512).to(self.device)
        
        return video_features
    
    def forward(self, frames):
        """Forward pass through the model"""
        visual_features = self.extract_visual_features(frames)
        # Ensure consistent dtype with the model
        visual_features = visual_features.to(dtype=next(self.anomaly_head.parameters()).dtype)
        anomaly_score = self.anomaly_head(visual_features)
        return anomaly_score
