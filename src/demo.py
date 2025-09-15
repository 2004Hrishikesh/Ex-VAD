import torch
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.enhanced_exvad_model import EnhancedExVADModel
from utils.video_utils import sample_frames

def run_demo(model_path="models_saved/best_model.pth", data_root="data"):
    """Run demonstration of trained model"""
    print("\n" + "="*60)
    print("Ex-VAD: Explainable Video Anomaly Detection Demo")
    print("="*60 + "\n")
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    else:
        device = torch.device('cpu')
        print("⚠️ CUDA not available, using CPU")
    
    # Load model
    print("\nLoading model...")
    model = EnhancedExVADModel(device)
    if os.path.exists(model_path):
        model_state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(model_state_dict, strict=False)  # Allow missing keys
        print(f"✅ Model loaded successfully!")
    else:
        print("❌ No trained model found. Please train first.")
        return
    
    model.eval()
    
    # Process test videos
    test_path = os.path.join(data_root, 'videos', 'test')
    if not os.path.exists(test_path):
        print(f"❌ Test directory not found at {test_path}")
        return
        
    results = []
    videos_processed = 0
    max_videos_per_category = 3  # Number of videos to process from each category
    
    print("\nProcessing test videos...")
    for category in ['normal', 'abuse', 'explosion']:
        category_path = os.path.join(test_path, category)
        if not os.path.exists(category_path):
            print(f"⚠️ {category} folder not found")
            continue
        
        print(f"\nProcessing {category} videos:")
        video_files = [f for f in os.listdir(category_path) if f.endswith(('.mp4', '.avi'))]
        
        for video_file in video_files[:max_videos_per_category]:
            video_path = os.path.join(category_path, video_file)
            print(f"\nAnalyzing: {video_file}")
            
            try:
                # Sample frames
                frames = sample_frames(video_path, num_frames=16, device=device)
                
                # Load threshold info
                threshold_path = os.path.join(os.path.dirname(model_path), 'threshold_info.json')
                if os.path.exists(threshold_path):
                    with open(threshold_path, 'r') as f:
                        threshold_info = json.load(f)
                    threshold = threshold_info['optimal_threshold']
                else:
                    threshold = 0.5  # fallback
                
                # Get prediction
                with torch.no_grad():
                    anomaly_score, explanation, captions = model(frames)
                    anomaly_prob = torch.sigmoid(anomaly_score).item()
                
                # Process results
                if isinstance(anomaly_score, torch.Tensor):
                    score = torch.sigmoid(anomaly_score).item()
                else:
                    score = 0.5
                
                prediction = "ANOMALY" if score > threshold else "NORMAL"
                confidence = score if score > 0.5 else 1 - score
                
                print(f"\n{'='*50}")
                print(f"Video: {os.path.basename(video_path)}")
                print(f"True Label: {category.upper()}")
                print(f"Prediction: {prediction} (Confidence: {confidence:.3f})")
                print(f"Explanation: {explanation}")
                print(f"Captions: {captions}")
                
                results.append({
                    'video': os.path.basename(video_path),
                    'true_label': category,
                    'prediction': prediction,
                    'confidence': confidence,
                    'explanation': explanation,
                    'captions': captions
                })
                
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
    
    # Save results
    output_dir = "../outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'demo_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎉 Demo completed! Results saved to outputs/demo_results.json")

if __name__ == "__main__":
    run_demo()
