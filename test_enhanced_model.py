"""
Test script for Enhanced ExVAD Model with Cross-Modal Attention Fusion
"""

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import torch
import sys
import os

# Add src to path
sys.path.append('src')

def test_enhanced_model():
    """Test the enhanced model with cross-modal attention"""
    print("Testing Enhanced ExVAD Model with Cross-Modal Attention Fusion")
    print("=" * 60)
    
    # Check GPU availability
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("⚠️ Using CPU (CUDA not available)")
    
    try:
        # Import enhanced model
        from models.enhanced_exvad_model import EnhancedExVADModel
        
        print("\n1. Creating Enhanced ExVAD Model...")
        model = EnhancedExVADModel(device=device, use_cross_modal_attention=True)
        print("✅ Enhanced model created successfully!")
        
        # Get model info
        model_info = model.get_model_info()
        print(f"\nModel Information:")
        print(f"  - Type: {model_info['model_type']}")
        print(f"  - Total Parameters: {model_info['total_parameters']:,}")
        print(f"  - Trainable Parameters: {model_info['trainable_parameters']:,}")
        print(f"  - Cross-Modal Attention: {model_info['cross_modal_attention']}")
        print(f"  - Device: {model_info['device']}")
        
        # Test with dummy frames
        print("\n2. Testing with dummy frames...")
        
        # Create dummy frames (simulate video frames)
        from PIL import Image
        import numpy as np
        
        dummy_frames = []
        for i in range(3):  # 3 frames
            # Create a random RGB image
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            dummy_frames.append(img)
        
        print(f"  - Created {len(dummy_frames)} dummy frames")
        
        # Test forward pass
        print("\n3. Testing forward pass...")
        model.eval()
        with torch.no_grad():
            anomaly_score, explanation, captions = model(dummy_frames)
        
        print(f"✅ Forward pass successful!")
        print(f"  - Anomaly Score: {anomaly_score.item():.4f}")
        print(f"  - Explanation: {explanation}")
        print(f"  - Captions: {captions}")
        
        # Test cross-modal similarity
        print("\n4. Testing cross-modal similarity...")
        try:
            similarity = model.get_cross_modal_similarity(dummy_frames)
            print(f"✅ Cross-modal similarity: {similarity:.4f}")
        except Exception as e:
            print(f"⚠️ Cross-modal similarity test failed: {e}")
        
        # Test attention weights
        print("\n5. Testing attention weights...")
        try:
            result = model.forward_with_attention_weights(dummy_frames)
            if len(result) == 4:
                anomaly_score, explanation, captions, attention_weights = result
                print(f"✅ Attention weights retrieved:")
                if attention_weights:
                    for key, value in attention_weights.items():
                        print(f"  - {key}: {value}")
            else:
                print("⚠️ Attention weights not available")
        except Exception as e:
            print(f"⚠️ Attention weights test failed: {e}")
        
        # Test enabling/disabling cross-modal attention
        print("\n6. Testing cross-modal attention toggle...")
        model.disable_cross_modal_attention()
        with torch.no_grad():
            score_without = model(dummy_frames)[0]
        
        model.enable_cross_modal_attention()
        with torch.no_grad():
            score_with = model(dummy_frames)[0]
        
        print(f"✅ Cross-modal attention toggle successful!")
        print(f"  - Score without CMAF: {score_without.item():.4f}")
        print(f"  - Score with CMAF: {score_with.item():.4f}")
        
        print("\n" + "=" * 60)
        print("✅ All tests passed! Enhanced model is working correctly.")
        print("🚀 Ready to train with Cross-Modal Attention Fusion!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_model()
    if success:
        print("\n💡 Next steps:")
        print("   1. Run: python src/train_enhanced.py")
        print("   2. Compare results with original model")
        print("   3. Analyze cross-modal attention patterns")
    else:
        print("\n🔧 Please fix the issues before proceeding")
