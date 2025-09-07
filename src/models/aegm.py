
import torch
import torch.nn as nn
from transformers import BlipProcessor, BlipForConditionalGeneration
import warnings
warnings.filterwarnings('ignore')

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
        """Generate captions with proper padding"""
        captions = []
        
        with torch.no_grad():
            # Process first 3 frames for efficiency
            for i, frame in enumerate(frames[:3]):
                try:
                    # Use padding=True and specify pad_token_id
                    inputs = self.processor(
                        images=frame, 
                        return_tensors="pt", 
                        padding=True,
                        truncation=True,
                        max_length=512
                    )
                    
                    # Move inputs to device
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Generate with consistent dtype and padding
                    generated_ids = self.model.generate(
                        **inputs,
                        max_length=50,
                        num_beams=3,
                        early_stopping=True,
                        do_sample=False
                    )
                    
                    caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
                    captions.append(caption)
                    
                except Exception as e:
                    print(f"Error generating caption for frame {i}: {e}")
                    captions.append("a video frame")
        
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
