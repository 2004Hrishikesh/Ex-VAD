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
warnings.filterwarnings('ignore')

from models.exvad_model import ExVADModel
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
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of GPU memory
        
        # Enable TensorFloat32 (TF32) for faster training
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        print("✅ GPU optimizations enabled")
        return device
    else:
        device = torch.device('cpu')
        print("⚠️ CUDA not available, using CPU")
        return device

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    train_pbar = tqdm(train_loader, desc="Training")
    
    for batch_idx, (frames_batch, labels, class_names, video_paths) in enumerate(train_pbar):
        # Get first sample from batch (batch_size=1)
        frames = frames_batch[0]
        label = labels[0].float().to(device)
        
        if len(frames) == 0:
            continue
        
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
        try:
            # Forward pass
            anomaly_score, explanation, captions = model(frames)
            
            # Ensure anomaly_score is on correct device and proper shape
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
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            with torch.no_grad():
                predicted = (torch.sigmoid(anomaly_score) > 0.5).float()
                target = label.unsqueeze(0) if label.dim() == 0 else label
                correct += (predicted == target).sum().item()
                total += target.size(0)
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{correct/total:.4f}' if total > 0 else '0.0000'
            })
            
            # Memory management
            if device.type == 'cuda' and batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"Error in training step: {e}")
            continue
    
    return total_loss / len(train_loader), correct / total if total > 0 else 0

def validate_epoch(model, test_loader, criterion, device, save_path=None, epoch=None):
    """Validate for one epoch with comprehensive metrics"""
    model.eval()
    total_loss = 0
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Validation")
        
        for frames_batch, labels, class_names, video_paths in test_pbar:
            # Get first sample from batch
            frames = frames_batch[0]
            label = labels[0].float().to(device)
            
            if len(frames) == 0:
                continue
            
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
    
    # Calculate comprehensive metrics
    from utils.evaluation_metrics import calculate_metrics
    metrics = calculate_metrics(all_labels, all_scores)
    
    # Plot metrics if last epoch
    if len(test_loader.dataset) > 0:
        from utils.evaluation_metrics import plot_metrics
        plot_metrics(metrics, save_path=os.path.join(save_path, f'metrics_epoch_{epoch}.png'))
    
    return total_loss / len(test_loader), metrics['accuracy']

def train_model(data_root="../data", epochs=5, learning_rate=1e-3, save_path="../models_saved"):
    """Main training function"""
    print("="*60)
    print("Ex-VAD Training with GPU Optimization")
    print("="*60)
    
    # Setup device
    device = setup_device()
    
    # Create model
    print("Loading model...")
    model = ExVADModel(device).to(device)
    
    # Create data loaders
    print("Loading datasets...")
    train_loader, test_loader = create_dataloaders(
        data_root='data',
        batch_size=1,
        num_frames=16,
        device=device
    )
    
    # Setup training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    
    # Import threshold tuning
    from utils.threshold_tuning import find_optimal_threshold
    
    # Training loop
    best_accuracy = 0
    training_history = []
    
    os.makedirs(save_path, exist_ok=True)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, test_loader, criterion, device, save_path, epoch)
        
        # Update scheduler
        scheduler.step()
        
        # Print results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
            print(f"✅ New best model saved! Accuracy: {best_accuracy:.4f}")
        
        # GPU memory info
        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory: {allocated:.2f}GB allocated / {reserved:.2f}GB reserved")
            torch.cuda.empty_cache()
        
        # Save training history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
    
    # Save training history
    with open(os.path.join(save_path, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\n🎉 Training completed! Best accuracy: {best_accuracy:.4f}")
    
    # Find optimal threshold
    print("\nFinding optimal threshold...")
    model.load_state_dict(torch.load(os.path.join(save_path, 'best_model.pth')))
    optimal_threshold, threshold_metrics = find_optimal_threshold(model, test_loader, device)
    
    # Save threshold and metrics
    threshold_info = {
        'optimal_threshold': optimal_threshold,
        'metrics': threshold_metrics
    }
    with open(os.path.join(save_path, 'threshold_info.json'), 'w') as f:
        json.dump(threshold_info, f, indent=2)
    
    print("✅ Optimal threshold saved!")
    return model, training_history, threshold_info

if __name__ == "__main__":
    # Test GPU first
    from test_gpu import test_gpu_setup
    
    if test_gpu_setup():
        print("\n" + "="*60)
        print("Starting training...")
        model, history = train_model(
            data_root="../data",
            epochs=5,
            learning_rate=1e-3,
            save_path="../models_saved"
        )
    else:
        print("❌ GPU setup failed. Please fix GPU issues before training.")
