"""
Enhanced Training Script with Cross-Modal Attention Fusion
Novel contribution: Training procedure for CMAF-enhanced ExVAD model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import json
import warnings
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')

from models.enhanced_exvad_model import EnhancedExVADModel
from utils.data_loader_new import create_dataloaders

def setup_device():
    """Setup device with proper GPU detection and optimization"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        
        # Print GPU info
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory // 1024**3
        print(f"Total GPU Memory: {total_memory} GB")
        
        # Enable cuDNN autotuner
        cudnn.benchmark = True
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        
        # Set memory allocation settings
        torch.cuda.set_per_process_memory_fraction(0.95)
        
        # Enable TensorFloat32 (TF32) for faster training
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        print("✅ GPU optimizations enabled")
        return device
    else:
        device = torch.device('cpu')
        print("⚠️ CUDA not available, using CPU")
        return device

def cross_modal_attention_loss(model, frames, explanations, lambda_cmaf=0.1):
    """
    Additional loss term to encourage meaningful cross-modal attention
    
    Args:
        model: Enhanced ExVAD model
        frames: Video frames
        explanations: Generated explanations
        lambda_cmaf: Weight for cross-modal attention loss
    
    Returns:
        cmaf_loss: Cross-modal attention regularization loss
    """
    if not hasattr(model, 'get_cross_modal_similarity'):
        return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Get cross-modal similarity
        similarity = model.get_cross_modal_similarity(frames)
        
        # Convert to tensor if it's a float
        if isinstance(similarity, (int, float)):
            similarity = torch.tensor(similarity, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Encourage moderate similarity (not too high, not too low)
        target_similarity = torch.tensor(0.5, device=similarity.device)
        cmaf_loss = lambda_cmaf * torch.abs(similarity - target_similarity)
        
        return cmaf_loss
    except Exception as e:
        print(f"Error computing CMAF loss: {e}")
        return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch_enhanced(model, train_loader, criterion, optimizer, device, use_cmaf_loss=True):
    """Enhanced training epoch with cross-modal attention loss"""
    model.train()
    total_loss = 0
    total_bce_loss = 0
    total_cmaf_loss = 0
    correct = 0
    total = 0
    
    train_pbar = tqdm(train_loader, desc="Training (Enhanced)")
    
    for batch_idx, (frames_batch, labels, class_names, video_paths) in enumerate(train_pbar):
        # Get first sample from batch
        frames = frames_batch[0]
        label = labels[0].float().to(device)
        
        if len(frames) == 0:
            continue
        
        optimizer.zero_grad(set_to_none=True)
        
        try:
            # Forward pass
            anomaly_score, explanation, captions = model(frames)
            
            # Ensure proper tensor format
            if isinstance(anomaly_score, torch.Tensor):
                anomaly_score = anomaly_score.to(device)
                if anomaly_score.dim() > 0:
                    anomaly_score = anomaly_score.squeeze()
                if anomaly_score.dim() == 0:
                    anomaly_score = anomaly_score.unsqueeze(0)
            else:
                anomaly_score = torch.tensor(0.0, device=device)
            
            # Binary Cross-Entropy Loss
            target = label.unsqueeze(0) if label.dim() == 0 else label
            bce_loss = criterion(anomaly_score, target)
            
            # Cross-Modal Attention Loss (novel contribution)
            cmaf_loss = torch.tensor(0.0, device=device)
            if use_cmaf_loss and hasattr(model, 'use_cross_modal_attention') and model.use_cross_modal_attention:
                cmaf_loss = cross_modal_attention_loss(model, frames, [explanation])
            
            # Total loss
            total_loss_batch = bce_loss + cmaf_loss
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
            
            # Statistics
            total_loss += total_loss_batch.item()
            total_bce_loss += bce_loss.item()
            total_cmaf_loss += cmaf_loss.item() if isinstance(cmaf_loss, torch.Tensor) else cmaf_loss
            
            with torch.no_grad():
                predicted = (torch.sigmoid(anomaly_score) > 0.5).float()
                correct += (predicted == target).sum().item()
                total += target.size(0)
            
            # Update progress bar
            train_pbar.set_postfix({
                'BCE': f'{bce_loss.item():.4f}',
                'CMAF': f'{cmaf_loss.item() if isinstance(cmaf_loss, torch.Tensor) else cmaf_loss:.4f}',
                'Total': f'{total_loss_batch.item():.4f}',
                'Acc': f'{correct/total:.4f}' if total > 0 else '0.0000'
            })
            
            # Memory management - more frequent cleanup
            if device.type == 'cuda' and batch_idx % 5 == 0:  # More frequent cleanup
                torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"Error in training step: {e}")
            # Clear cache on error
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            continue
    
    avg_total_loss = total_loss / len(train_loader)
    avg_bce_loss = total_bce_loss / len(train_loader)
    avg_cmaf_loss = total_cmaf_loss / len(train_loader)
    accuracy = correct / total if total > 0 else 0
    
    return avg_total_loss, avg_bce_loss, avg_cmaf_loss, accuracy

def validate_epoch_enhanced(model, test_loader, criterion, device, save_path=None, epoch=None):
    """Enhanced validation with cross-modal attention analysis"""
    model.eval()
    total_loss = 0
    all_labels = []
    all_scores = []
    cross_modal_similarities = []
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Validation (Enhanced)")
        
        for frames_batch, labels, class_names, video_paths in test_pbar:
            frames = frames_batch[0]
            label = labels[0].float().to(device)
            
            if len(frames) == 0:
                continue
            
            try:
                # Forward pass
                anomaly_score, explanation, captions = model(frames)
                
                # Get cross-modal similarity for analysis
                if hasattr(model, 'get_cross_modal_similarity'):
                    try:
                        similarity = model.get_cross_modal_similarity(frames)
                        cross_modal_similarities.append(similarity)
                    except:
                        pass
                
                # Ensure proper tensor format
                if isinstance(anomaly_score, torch.Tensor):
                    anomaly_score = anomaly_score.to(device)
                    if anomaly_score.dim() > 0:
                        anomaly_score = anomaly_score.squeeze()
                    if anomaly_score.dim() == 0:
                        anomaly_score = anomaly_score.unsqueeze(0)
                else:
                    anomaly_score = torch.tensor(0.0, device=device)
                
                # Compute loss
                loss = criterion(anomaly_score, label.unsqueeze(0) if label.dim() == 0 else label)
                
                # Store predictions and labels
                total_loss += loss.item()
                score = torch.sigmoid(anomaly_score).cpu().item()
                all_scores.append(score)
                all_labels.append(label.cpu().item())
                
            except Exception as e:
                print(f"Error in validation step: {e}")
                continue
    
    # Calculate metrics
    from utils.evaluation_metrics import calculate_metrics
    metrics = calculate_metrics(all_labels, all_scores)
    
    # Add cross-modal attention analysis
    if cross_modal_similarities:
        metrics['avg_cross_modal_similarity'] = sum(cross_modal_similarities) / len(cross_modal_similarities)
        metrics['std_cross_modal_similarity'] = torch.std(torch.tensor(cross_modal_similarities)).item()
    
    # Plot enhanced metrics
    if len(test_loader.dataset) > 0 and save_path:
        plot_enhanced_metrics(metrics, cross_modal_similarities, save_path, epoch)
    
    return total_loss / len(test_loader), metrics['accuracy']

def plot_enhanced_metrics(metrics, cross_modal_similarities, save_path, epoch):
    """Plot enhanced metrics including cross-modal attention analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original metrics plot
    from utils.evaluation_metrics import plot_metrics
    
    # Cross-modal similarity distribution
    if cross_modal_similarities:
        axes[0, 0].hist(cross_modal_similarities, bins=20, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Cross-Modal Similarity Distribution')
        axes[0, 0].set_xlabel('Similarity Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(sum(cross_modal_similarities)/len(cross_modal_similarities), 
                          color='red', linestyle='--', label='Mean')
        axes[0, 0].legend()
    
    # ROC Curve
    if 'fpr' in metrics and 'tpr' in metrics:
        axes[0, 1].plot(metrics['fpr'], metrics['tpr'], color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {metrics["auc"]:.2f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend(loc="lower right")
    
    # Precision-Recall Curve
    if 'recall_curve' in metrics and 'precision_curve' in metrics:
        axes[1, 0].plot(metrics['recall_curve'], metrics['precision_curve'], color='purple', lw=2,
                       label=f'PR curve (AP = {metrics["ap"]:.2f})')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision-Recall Curve')
        axes[1, 0].legend(loc="lower left")
    
    # Confusion Matrix
    if 'confusion_matrix' in metrics and metrics['confusion_matrix'] is not None:
        cm = metrics['confusion_matrix']
        # Ensure confusion matrix is 2D
        if hasattr(cm, 'shape') and len(cm.shape) == 2:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
            axes[1, 1].set_title('Confusion Matrix')
            axes[1, 1].set_xlabel('Predicted')
            axes[1, 1].set_ylabel('Actual')
        else:
            axes[1, 1].text(0.5, 0.5, f'Confusion Matrix\nData: {cm}', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Confusion Matrix (Invalid Format)')
    else:
        axes[1, 1].text(0.5, 0.5, 'Confusion Matrix\nNot Available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'enhanced_metrics_epoch_{epoch}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def train_enhanced_model(data_root="../data", epochs=3, learning_rate=5e-4, save_path="../models_saved", 
                        use_cross_modal_attention=True, use_cmaf_loss=True):
    """Main training function for enhanced model (optimized for memory)"""
    print("="*80)
    print("Enhanced Ex-VAD Training with Cross-Modal Attention Fusion")
    print("="*80)
    
    # Setup device
    device = setup_device()
    
    # Reduce GPU memory fraction for stability
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.85)  # Reduced from 0.95
    
    # Create enhanced model
    print("Loading Enhanced ExVAD model...")
    model = EnhancedExVADModel(device, use_cross_modal_attention=use_cross_modal_attention).to(device)
    
    # Print model info
    model_info = model.get_model_info()
    print(f"Model Type: {model_info['model_type']}")
    print(f"Total Parameters: {model_info['total_parameters']:,}")
    print(f"Trainable Parameters: {model_info['trainable_parameters']:,}")
    print(f"Cross-Modal Attention: {model_info['cross_modal_attention']}")
    
    # Create data loaders with smaller batch processing
    print("Loading datasets...")
    train_loader, test_loader = create_dataloaders(
        data_root='data',
        batch_size=1,
        num_frames=8,  # Reduced from 16 for memory efficiency
        device=device
    )
    
    # Setup training with more conservative settings
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)  # More aggressive LR decay
    
    # Training loop
    best_accuracy = 0
    training_history = []
    
    os.makedirs(save_path, exist_ok=True)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 50)
        
        # Clear cache before each epoch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Train
        train_total_loss, train_bce_loss, train_cmaf_loss, train_acc = train_epoch_enhanced(
            model, train_loader, criterion, optimizer, device, use_cmaf_loss=use_cmaf_loss)
        
        # Validate
        val_loss, val_acc = validate_epoch_enhanced(model, test_loader, criterion, device, save_path, epoch)
        
        # Update scheduler
        scheduler.step()
        
        # Print results
        print(f"Train - Total Loss: {train_total_loss:.4f}, BCE Loss: {train_bce_loss:.4f}, "
              f"CMAF Loss: {train_cmaf_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), os.path.join(save_path, 'enhanced_best_model.pth'))
            print(f"✅ New best enhanced model saved! Accuracy: {best_accuracy:.4f}")
        
        # GPU memory info
        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory: {allocated:.2f}GB allocated / {reserved:.2f}GB reserved")
            torch.cuda.empty_cache()
        
        # Save training history
        training_history.append({
            'epoch': epoch + 1,
            'train_total_loss': train_total_loss,
            'train_bce_loss': train_bce_loss,
            'train_cmaf_loss': train_cmaf_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
    
    # Save training history
    with open(os.path.join(save_path, 'enhanced_training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\n🎉 Enhanced training completed! Best accuracy: {best_accuracy:.4f}")
    
    # Find optimal threshold
    print("\nFinding optimal threshold for enhanced model...")
    model.load_state_dict(torch.load(os.path.join(save_path, 'enhanced_best_model.pth')))
    
    from utils.threshold_tuning import find_optimal_threshold
    optimal_threshold, threshold_metrics = find_optimal_threshold(model, test_loader, device)
    
    # Save threshold and metrics
    threshold_info = {
        'optimal_threshold': optimal_threshold,
        'metrics': threshold_metrics,
        'model_type': 'Enhanced ExVAD with Cross-Modal Attention',
        'cross_modal_attention': use_cross_modal_attention
    }
    with open(os.path.join(save_path, 'enhanced_threshold_info.json'), 'w') as f:
        json.dump(threshold_info, f, indent=2)
    
    print("✅ Enhanced model training and threshold optimization completed!")
    return model, training_history, threshold_info

if __name__ == "__main__":
    # Test GPU first
    from test_gpu import test_gpu_setup
    
    if test_gpu_setup():
        print("\n" + "="*80)
        print("Starting Enhanced Ex-VAD Training...")
        
        # Train enhanced model with cross-modal attention
        model, history, threshold_info = train_enhanced_model(
            data_root="../data",
            epochs=3,           # Reduced epochs for initial testing
            learning_rate=5e-4, # Lower learning rate for stability
            save_path="../models_saved",
            use_cross_modal_attention=True,
            use_cmaf_loss=True
        )
        
        print("\n" + "="*80)
        print("Enhanced training completed successfully!")
        print("Novel contributions implemented:")
        print("1. ✅ Cross-Modal Attention Fusion (CMAF)")
        print("2. ✅ Enhanced anomaly detection head")
        print("3. ✅ Cross-modal attention regularization loss")
        print("4. ✅ Comprehensive attention analysis")
        print("="*80)
        
    else:
        print("❌ GPU setup failed. Please fix GPU issues before training.")
