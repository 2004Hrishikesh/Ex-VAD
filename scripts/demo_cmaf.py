"""
Demo Script for CMAF-Enhanced ExVAD Model
Tests one video from each category and provides detailed accuracy metrics
"""
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
from datetime import datetime
from src.models.enhanced_exvad_model import EnhancedExVADModel
from src.utils.video_utils import sample_frames, frames_to_tensor
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class CMAPModelDemo:
    def __init__(self, model_path="models_saved/best_model.pth", 
                 data_root="data/videos/test",
                 optimal_threshold=0.1):
        self.model_path = model_path
        self.data_root = data_root
        self.optimal_threshold = optimal_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Define all categories
        self.categories = {
            'NormalVideos': {'expected': 'normal', 'label': 0, 'description': 'Normal Activity'},
            'Robbery': {'expected': 'anomaly', 'label': 1, 'description': 'Robbery in progress'},
            'Fighting': {'expected': 'anomaly', 'label': 1, 'description': 'Physical altercation'},
            'Shooting': {'expected': 'anomaly', 'label': 1, 'description': 'Gun shooting incident'},
            'Shoplifting': {'expected': 'anomaly', 'label': 1, 'description': 'Theft from store'},
            'Assault': {'expected': 'anomaly', 'label': 1, 'description': 'Physical assault'},
            'Burglary': {'expected': 'anomaly', 'label': 1, 'description': 'Breaking and entering'},
            'Arson': {'expected': 'anomaly', 'label': 1, 'description': 'Fire setting'},
            'Vandalism': {'expected': 'anomaly', 'label': 1, 'description': 'Property damage'},
            'Explosion': {'expected': 'anomaly', 'label': 1, 'description': 'Explosion event'},
            'Arrest': {'expected': 'anomaly', 'label': 1, 'description': 'Police arrest'},
            'Abuse': {'expected': 'anomaly', 'label': 1, 'description': 'Abusive behavior'},
            'RoadAccidents': {'expected': 'anomaly', 'label': 1, 'description': 'Vehicle accident'},
            'Stealing': {'expected': 'anomaly', 'label': 1, 'description': 'Theft incident'}
        }
        
        self.results = []
        self.model = None
    
    def load_model(self):
        """Load the CMAF-enhanced model"""
        print("\n" + "="*80)
        print("CMAF-ENHANCED EXVAD MODEL DEMO")
        print("="*80)
        print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f" Device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
        
        print(f"\nLoading model from: {self.model_path}")
        
        self.model = EnhancedExVADModel(self.device)
        
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Check if this is a CMAF-trained model
            is_cmaf_model = False
            if isinstance(checkpoint, dict):
                if 'architecture' in checkpoint:
                    is_cmaf_model = 'CMAF' in checkpoint.get('architecture', '')
                    print(f"Architecture: {checkpoint['architecture']}")
                if 'epoch' in checkpoint:
                    print(f"Epoch: {checkpoint['epoch']}")
                if 'best_accuracy' in checkpoint:
                    print(f"Training Accuracy: {checkpoint['best_accuracy']:.4f}")
                
                # Load state dict
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    self.model.load_state_dict(checkpoint, strict=False)
            else:
                # Old model format
                self.model.load_state_dict(checkpoint, strict=False)
                print(" OLD model format detected (no CMAF training metadata)")
                
                # Verify CMAF components
                state_keys = list(checkpoint.keys())
                has_cmaf = any('visual_encoder' in k or 'cross_modal' in k for k in state_keys)
                is_cmaf_model = has_cmaf
                
                if not is_cmaf_model:
                    print("WARNING: Model lacks CMAF components!")
                    print("   Train with: python src/train_enhanced.py")
            
            if is_cmaf_model:
                print("CMAF-Enhanced model loaded!")
            else:
                print(" Model loaded (OLD architecture)")
        else:
            print(" Warning: No trained model found, using untrained model")
        
        self.model.eval()
        print(f"Detection threshold: {self.optimal_threshold}")
    
    def get_videos_from_category(self, category, num_videos=3):
        """Collect sample descriptors (video files or frame sequences) for a category"""
        category_path = os.path.join(self.data_root, category)
        if not os.path.exists(category_path):
            return []

        # Prefer direct video files when available
        video_files = sorted([
            f for f in os.listdir(category_path)
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        ])

        samples = []
        for filename in video_files[:num_videos]:
            samples.append({
                'type': 'video',
                'path': os.path.join(category_path, filename),
                'display_name': filename
            })

        if samples:
            return samples

        # Fall back to PNG frame sequences (e.g., Robbery001_x264_0.png ...)
        frame_groups = self._group_frame_sequences(category_path)
        for base_name, frame_files in frame_groups.items():
            samples.append({
                'type': 'frame_sequence',
                'category_path': category_path,
                'frame_files': frame_files,
                'display_name': f"{base_name} (frames)"
            })
            if len(samples) >= num_videos:
                break

        return samples

    @staticmethod
    def _extract_sequence_base(frame_file):
        """Derive the logical sequence name from a frame filename"""
        name, _ = os.path.splitext(frame_file)
        parts = name.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
        return name

    def _group_frame_sequences(self, category_path):
        """Group PNG frames into sequences keyed by their base name"""
        frame_files = sorted([
            f for f in os.listdir(category_path)
            if f.lower().endswith('.png')
        ])

        groups = {}
        for frame_file in frame_files:
            base_name = self._extract_sequence_base(frame_file)
            groups.setdefault(base_name, []).append(frame_file)

        # Ensure frames are sorted numerically where possible
        for base_name, files in groups.items():
            groups[base_name] = sorted(
                files,
                key=lambda f: int(os.path.splitext(f)[0].rsplit('_', 1)[1])
                if os.path.splitext(f)[0].rsplit('_', 1)[-1].isdigit()
                else f
            )

        return groups

    def _load_frame_sequence(self, category_path, frame_files, num_frames=16):
        """Load a fixed number of frames from a PNG sequence"""
        if not frame_files:
            raise ValueError("No frame files provided for sequence loading")

        frame_paths = [os.path.join(category_path, f) for f in frame_files]
        total_frames = len(frame_paths)

        if total_frames >= num_frames:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        else:
            indices = list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)

        frames = []
        for idx in indices:
            frame_path = frame_paths[idx]
            try:
                frame = Image.open(frame_path).convert('RGB')
                frames.append(frame)
            except Exception as exc:
                print(f"   Error loading frame {frame_path}: {exc}")
                break

        if not frames:
            raise ValueError("Failed to load any frames from sequence")

        # Pad with last frame if we hit an exception early
        while len(frames) < num_frames:
            frames.append(frames[-1])

        return frames[:num_frames]
    
    def test_video(self, sample, category_info):
        """Test a single sample (video file or frame sequence) and return results"""
        try:
            if sample['type'] == 'video':
                if not os.path.exists(sample['path']):
                    return None
                frames = sample_frames(sample['path'], num_frames=16)
            elif sample['type'] == 'frame_sequence':
                frames = self._load_frame_sequence(sample['category_path'], sample['frame_files'], num_frames=16)
            else:
                print(f"   Unsupported sample type: {sample['type']}")
                return None

            frames_tensor = frames_to_tensor(frames, self.device).unsqueeze(0)

            with torch.no_grad():
                anomaly_score, explanation, captions = self.model(frames_tensor)

            score = anomaly_score.item() if torch.is_tensor(anomaly_score) else anomaly_score
            predicted_label = 1 if score > self.optimal_threshold else 0
            is_correct = predicted_label == category_info['label']

            return {
                'video': sample.get('display_name', 'sample'),
                'path': sample.get('path', sample.get('category_path')),
                'source_type': sample['type'],
                'anomaly_score': score,
                'predicted_label': predicted_label,
                'expected_label': category_info['label'],
                'is_correct': is_correct,
                'explanation': explanation,
                'captions': captions,
                'category_description': category_info['description']
            }

        except Exception as e:
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_demo(self, videos_per_category=3):
        """Run the demo on multiple videos per category"""
        self.load_model()
        
        print("\n" + "="*80)
        print(f"TESTING {videos_per_category} VIDEO(S) PER CATEGORY")
        print("="*80)
        
        total_correct = 0
        total_tested = 0
        normal_correct = 0
        normal_total = 0
        anomaly_correct = 0
        anomaly_total = 0
        
        for category, info in self.categories.items():
            print(f"\n{'='*80}")
            print(f"Category: {category.upper()}")
            print(f"Description: {info['description']}")
            print(f"Expected: {info['expected'].upper()}")
            print(f"{'='*80}")
            
            # Get multiple videos from category
            samples = self.get_videos_from_category(category, videos_per_category)
            
            if not samples:
                print(f" No videos or frame sequences found in {category}, skipping...")
                continue
            
            print(f"Testing {len(samples)} sample(s) from {category}...")
            
            category_correct = 0
            category_total = 0
            
            # Test each video in the category
            for idx, sample in enumerate(samples, 1):
                display_name = sample.get('display_name', os.path.basename(sample.get('path', 'sample')))
                print(f"\n  Sample {idx}/{len(samples)}: {display_name}")
                
                # Test the video
                result = self.test_video(sample, info)
                
                if result is None:
                    continue
                
                # Store result
                result['category'] = category
                result['expected_type'] = info['expected']
                result['video_number'] = idx
                self.results.append(result)
                
                # Display results
                print(f"     Score: {result['anomaly_score']:.4f}")
                print(f"     Predicted: {' ANOMALY' if result['predicted_label'] == 1 else 'NORMAL'}")
                print(f"     Expected: {' ANOMALY' if result['expected_label'] == 1 else 'NORMAL'}")
                
                # Check correctness
                if result['is_correct']:
                    print(f"     Result: CORRECT")
                    total_correct += 1
                    category_correct += 1
                else:
                    print(f"     Result: INCORRECT")
                
                total_tested += 1
                category_total += 1
                
                # Track normal vs anomaly accuracy
                if info['expected'] == 'normal':
                    normal_total += 1
                    if result['is_correct']:
                        normal_correct += 1
                else:
                    anomaly_total += 1
                    if result['is_correct']:
                        anomaly_correct += 1
                
                # Display explanation (compact format)
                print(f"     {result['explanation'][:80]}...")
            
            # Category summary
            if category_total > 0:
                category_acc = (category_correct / category_total) * 100
                print(f"\n  � {category} Summary: {category_acc:.1f}% ({category_correct}/{category_total})")
        
        # Calculate final metrics
        self.display_final_metrics(total_correct, total_tested, 
                                   normal_correct, normal_total, 
                                   anomaly_correct, anomaly_total)
        
        # Save results
        self.save_results()
        
        # Generate visualization
        self.generate_visualization()
    
    def display_final_metrics(self, total_correct, total_tested, 
                             normal_correct, normal_total, 
                             anomaly_correct, anomaly_total):
        """Display comprehensive accuracy metrics"""
        print("\n" + "="*80)
        print("FINAL ACCURACY METRICS")
        print("="*80)
        
        if total_tested > 0:
            overall_accuracy = (total_correct / total_tested) * 100
            print(f"\nOverall Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_tested})")
        else:
            print("\n No videos were tested")
            return
        
        # Normal videos accuracy
        if normal_total > 0:
            normal_accuracy = (normal_correct / normal_total) * 100
            print(f"\nNormal Videos Accuracy: {normal_accuracy:.2f}% ({normal_correct}/{normal_total})")
        
        # Anomaly videos accuracy
        if anomaly_total > 0:
            anomaly_accuracy = (anomaly_correct / anomaly_total) * 100
            print(f" Anomaly Videos Accuracy: {anomaly_accuracy:.2f}% ({anomaly_correct}/{anomaly_total})")
        
        # Calculate precision, recall, F1
        true_positives = sum(1 for r in self.results if r['predicted_label'] == 1 and r['expected_label'] == 1)
        false_positives = sum(1 for r in self.results if r['predicted_label'] == 1 and r['expected_label'] == 0)
        false_negatives = sum(1 for r in self.results if r['predicted_label'] == 0 and r['expected_label'] == 1)
        true_negatives = sum(1 for r in self.results if r['predicted_label'] == 0 and r['expected_label'] == 0)
        
        if true_positives + false_positives > 0:
            precision = true_positives / (true_positives + false_positives)
            print(f"\nPrecision: {precision:.4f}")
        
        if true_positives + false_negatives > 0:
            recall = true_positives / (true_positives + false_negatives)
            print(f"Recall: {recall:.4f}")
        
        if true_positives + false_positives > 0 and true_positives + false_negatives > 0:
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            if precision + recall > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
                print(f"F1 Score: {f1_score:.4f}")
        
        # Confusion matrix
        print(f"\nConfusion Matrix:")
        print(f"   True Positives (Anomaly detected correctly): {true_positives}")
        print(f"   True Negatives (Normal detected correctly): {true_negatives}")
        print(f"   False Positives (Normal classified as Anomaly): {false_positives}")
        print(f"   False Negatives (Anomaly classified as Normal): {false_negatives}")
        
        # Per-category breakdown
        print(f"\nPer-Category Results:")
        print(f"{'Category':<20} {'Score':<10} {'Predicted':<12} {'Expected':<12} {'Result':<10}")
        print("-" * 80)
        
        for result in self.results:
            category = result['category']
            score = result['anomaly_score']
            predicted = 'ANOMALY' if result['predicted_label'] == 1 else 'NORMAL'
            expected = 'ANOMALY' if result['expected_label'] == 1 else 'NORMAL'
            status = 'CORRECT' if result['is_correct'] else 'WRONG'
            
            print(f"{category:<20} {score:<10.4f} {predicted:<12} {expected:<12} {status:<10}")
    
    def save_results(self):
        """Save results to JSON file"""
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, "demo_results.json")
        
        # Calculate metrics
        total_tested = len(self.results)
        total_correct = sum(1 for r in self.results if r['is_correct'])
        overall_accuracy = (total_correct / total_tested * 100) if total_tested > 0 else 0
        
        normal_results = [r for r in self.results if r['expected_type'] == 'normal']
        anomaly_results = [r for r in self.results if r['expected_type'] == 'anomaly']
        
        normal_accuracy = (sum(1 for r in normal_results if r['is_correct']) / len(normal_results) * 100) if normal_results else 0
        anomaly_accuracy = (sum(1 for r in anomaly_results if r['is_correct']) / len(anomaly_results) * 100) if anomaly_results else 0
        
        output_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': self.model_path,
            'threshold': self.optimal_threshold,
            'device': str(self.device),
            'metrics': {
                'overall_accuracy': overall_accuracy,
                'total_correct': total_correct,
                'total_tested': total_tested,
                'normal_accuracy': normal_accuracy,
                'normal_tested': len(normal_results),
                'anomaly_accuracy': anomaly_accuracy,
                'anomaly_tested': len(anomaly_results)
            },
            'results': self.results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
    
    def generate_visualization(self):
        """Generate visualization of results"""
        if not self.results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Anomaly scores by category
        categories = [r['category'] for r in self.results]
        scores = [r['anomaly_score'] for r in self.results]
        colors = ['green' if r['is_correct'] else 'red' for r in self.results]
        
        axes[0, 0].barh(categories, scores, color=colors, alpha=0.7)
        axes[0, 0].axvline(x=self.optimal_threshold, color='blue', linestyle='--', 
                          linewidth=2, label=f'Threshold ({self.optimal_threshold})')
        axes[0, 0].set_xlabel('Anomaly Score')
        axes[0, 0].set_title('Anomaly Scores by Category')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # 2. Accuracy by type (Normal vs Anomaly)
        normal_results = [r for r in self.results if r['expected_type'] == 'normal']
        anomaly_results = [r for r in self.results if r['expected_type'] == 'anomaly']
        
        normal_acc = (sum(1 for r in normal_results if r['is_correct']) / len(normal_results) * 100) if normal_results else 0
        anomaly_acc = (sum(1 for r in anomaly_results if r['is_correct']) / len(anomaly_results) * 100) if anomaly_results else 0
        
        types = ['Normal', 'Anomaly']
        accuracies = [normal_acc, anomaly_acc]
        colors_acc = ['green', 'orange']
        
        bars = axes[0, 1].bar(types, accuracies, color=colors_acc, alpha=0.7)
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Accuracy by Video Type')
        axes[0, 1].set_ylim([0, 105])
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. Confusion Matrix
        true_positives = sum(1 for r in self.results if r['predicted_label'] == 1 and r['expected_label'] == 1)
        false_positives = sum(1 for r in self.results if r['predicted_label'] == 1 and r['expected_label'] == 0)
        false_negatives = sum(1 for r in self.results if r['predicted_label'] == 0 and r['expected_label'] == 1)
        true_negatives = sum(1 for r in self.results if r['predicted_label'] == 0 and r['expected_label'] == 0)
        
        confusion_matrix = np.array([[true_negatives, false_positives],
                                     [false_negatives, true_positives]])
        
        im = axes[1, 0].imshow(confusion_matrix, cmap='Blues', alpha=0.7)
        axes[1, 0].set_xticks([0, 1])
        axes[1, 0].set_yticks([0, 1])
        axes[1, 0].set_xticklabels(['Normal', 'Anomaly'])
        axes[1, 0].set_yticklabels(['Normal', 'Anomaly'])
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        axes[1, 0].set_title('Confusion Matrix')
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = axes[1, 0].text(j, i, confusion_matrix[i, j],
                                      ha="center", va="center", color="black",
                                      fontsize=20, fontweight='bold')
        
        plt.colorbar(im, ax=axes[1, 0])
        
        # 4. Overall metrics summary
        axes[1, 1].axis('off')
        
        total_tested = len(self.results)
        total_correct = sum(1 for r in self.results if r['is_correct'])
        overall_accuracy = (total_correct / total_tested * 100) if total_tested > 0 else 0
        
        if true_positives + false_positives > 0:
            precision = true_positives / (true_positives + false_positives)
        else:
            precision = 0
        
        if true_positives + false_negatives > 0:
            recall = true_positives / (true_positives + false_negatives)
        else:
            recall = 0
        
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0
        
        summary_text = f"""
        CMAF Model Performance Summary
        ================================
        
        Overall Accuracy: {overall_accuracy:.2f}%
        Total Videos Tested: {total_tested}
        Correct Predictions: {total_correct}
        
        Normal Videos Accuracy: {normal_acc:.2f}%
        Anomaly Videos Accuracy: {anomaly_acc:.2f}%
        
        Precision: {precision:.4f}
        Recall: {recall:.4f}
        F1 Score: {f1_score:.4f}
        
        Threshold: {self.optimal_threshold}
        Device: {self.device}
        """
        
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                       verticalalignment='center')
        
        plt.tight_layout()
        
        # Save figure
        output_file = "outputs/demo_visualization.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_file}")
        
        plt.close()

def main():
    """Main function to run the demo"""
    import sys
    
    # Get number of videos per category from command line or use default
    videos_per_category = 3  # Default: test 3 videos per category
    if len(sys.argv) > 1:
        try:
            videos_per_category = int(sys.argv[1])
        except:
            print(f"Invalid argument, using default: {videos_per_category} videos per category")
    
    print(f"\nTesting {videos_per_category} video(s) per category...")
    
    demo = CMAPModelDemo(
        model_path="models_saved/best_model.pth",
        data_root="data/Test",
        optimal_threshold=0.1
    )
    
    demo.run_demo(videos_per_category=videos_per_category)
    
    print("\n" + "="*80)
    print("DEMO COMPLETED SUCCESSFULLY! ")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
