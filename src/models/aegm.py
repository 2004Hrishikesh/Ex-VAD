
import torch
import torch.nn as nn
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')

class SimpleAEGM(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        print(f"Loading BLIP model on {device}...")
        
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32
        ).to(device)
        
        # Set pad_token if not available
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        
        self.model.eval()
        print("✅ BLIP model loaded successfully!")
    
    def generate_captions(self, frames):
        """Generate captions with proper padding and memory optimization"""
        captions = []
        
        with torch.no_grad():
            # Process only first frame for efficiency (reduce memory usage)
            for i, frame in enumerate(frames[:1]):  # Only process first frame
                try:
                    # Ensure frame is PIL Image
                    if not hasattr(frame, 'mode'):
                        if torch.is_tensor(frame):
                            frame_array = frame.cpu().numpy()
                            if frame_array.dtype != np.uint8:
                                frame_array = (frame_array * 255).astype(np.uint8)
                        else:
                            frame_array = frame
                        
                        if isinstance(frame_array, np.ndarray):
                            frame = Image.fromarray(frame_array)
                        else:
                            frame = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
                    
                    if frame.mode != 'RGB':
                        frame = frame.convert('RGB')
                    
                    # Use simpler processing to avoid conflicts
                    inputs = self.processor(
                        images=frame, 
                        return_tensors="pt"
                    )
                    
                    # Move inputs to device
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Generate with minimal settings to avoid parameter conflicts
                    generated_ids = self.model.generate(
                        **inputs,
                        max_length=25,
                        num_beams=1,  # Use greedy decoding instead of beam search
                        do_sample=False,
                        early_stopping=True
                    )
                    
                    caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
                    captions.append(caption)
                    
                    # Clear GPU cache after each generation
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"Error generating caption for frame {i}: {e}")
                    captions.append("a video frame")
                    # Clear cache on error too
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
        
        return captions if captions else ["a video frame"]
    
    def generate_explanation(self, captions):
        """Generate explanation from captions"""
        if not captions:
            return "Video shows: a scene"
        
        main_caption = captions[0]
        if len(captions) > 1:
            explanation = f"Video shows: {main_caption}. Also visible: {captions[1]}"
        else:
            explanation = f"Video shows: {main_caption}"
        
        return explanation
    
    def forward(self, frames):
        captions = self.generate_captions(frames)
        explanation = self.generate_explanation(captions)
        return explanation, captions

print("✅ Fixed AEGM model created!")
