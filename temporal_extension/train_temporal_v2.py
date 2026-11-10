"""
Training Script for Temporal-Enhanced ExVAD Model
Trains the model with temporal modeling components for improved video anomaly detection.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
import argparse
from datetime import datetime
from tqdm import tqdm
import numpy as np

# Add src to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from temporal_extension.models.temporal_enhanced_exvad_v2 import TemporalEnhancedExVAD
from src.utils.data_loader_new import VideoAnomalyDataset


def train_temporal_model(
    epochs=10,
    batch_size=1,
    learning_rate=1e-4,
    temporal_weight=0.3,
    use_transformer=True,
    use_lstm=True,
    use_motion=True,
    freeze_base=False,
    save_dir="models_saved",
    data_root="data"
):
    """
    Train the temporal-enhanced ExVAD model.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size (recommend 1 due to memory)
        learning_rate: Learning rate
        temporal_weight: Weight for temporal score in final prediction
        use_transformer: Whether to use temporal transformer
        use_lstm: Whether to use temporal LSTM
        use_motion: Whether to use motion encoding
        freeze_base: Whether to freeze base CMAF model
        save_dir: Directory to save models
        data_root: Root directory for data
    """
    print("\n" + "="*80)
    print("TEMPORAL-ENHANCED EXVAD TRAINING")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    
    # Training configuration
    config = {
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'temporal_weight': temporal_weight,
        'use_transformer': use_transformer,
        'use_lstm': use_lstm,
        'use_motion': use_motion,
        'freeze_base': freeze_base,
        'architecture': 'Temporal-Enhanced-CMAF-ExVAD'
    }
    
    print("\nConfiguration:")
    for k, v in config.items():
        print(f"   {k}: {v}")
    
    # Initialize model
    print("\nInitializing model...")
    model = TemporalEnhancedExVAD(
        device=device,
        temporal_hidden_dim=256,  # Reduced for memory efficiency
        use_temporal_transformer=use_transformer,
        use_temporal_lstm=use_lstm,
        use_motion_encoding=use_motion,
        temporal_weight=temporal_weight
    )
    
    if freeze_base:
        model._freeze_base_model()
    
    # Load base model weights if available
    base_checkpoint_path = os.path.join(save_dir, "best_model.pth")
    if os.path.exists(base_checkpoint_path):
        print(f"\nLoading base model weights from: {base_checkpoint_path}")
        try:
            checkpoint = torch.load(base_checkpoint_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.base_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.base_model.load_state_dict(checkpoint, strict=False)
            print("Base model weights loaded")
        except Exception as e:
            print(f" Could not load base weights: {e}")
    
    # Setup data
    print("\nLoading datasets...")
    
    # Use fewer frames and limit dataset size for faster training
    train_dataset = VideoAnomalyDataset(data_root=data_root, split='train', num_frames=8, device=device, max_samples_per_category=20)
    test_dataset = VideoAnomalyDataset(data_root=data_root, split='test', num_frames=8, device=device)
    
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    
    # Custom collate for list of frames
    def custom_collate(batch):
        frames_batch = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch], dtype=torch.float32)
        return frames_batch, labels
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=custom_collate
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate
    )
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    
    # Only train temporal components if base is frozen
    if freeze_base:
        params_to_train = [p for n, p in model.named_parameters() 
                          if 'base_model' not in n and p.requires_grad]
    else:
        params_to_train = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = optim.AdamW(params_to_train, lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"\nTrainable parameters: {sum(p.numel() for p in params_to_train):,}")
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'temporal_contribution': []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    
    # Training loop
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 50)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        temporal_scores = []
        
        train_pbar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, (frames_batch, labels) in enumerate(train_pbar):
            labels = labels.float().unsqueeze(1).to(device)
            
            # Clear GPU cache periodically
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            optimizer.zero_grad()
            
            try:
                # Forward pass - process one sample at a time
                frames = frames_batch[0]  # Get first (only) sample
                
                output = model(frames, return_details=True)
                score, explanation, captions, details = output
                
                # Loss
                loss = criterion(score, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(params_to_train, max_norm=1.0)
                
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                predicted = (score > 0.5).float()
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)
                
                # Track temporal contribution
                temporal_scores.append(details['temporal_score'].item())
                
                # Update progress bar
                train_pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100*train_correct/train_total:.1f}%"
                })
                
            except Exception as e:
                print(f"\n Error in batch {batch_idx}: {e}")
                continue
        
        # Epoch training stats
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = train_correct / train_total
        avg_temporal = np.mean(temporal_scores) if temporal_scores else 0
        
        print(f"\n   Training Loss: {epoch_train_loss:.4f}")
        print(f"   Training Accuracy: {epoch_train_acc*100:.2f}%")
        print(f"   Avg Temporal Score: {avg_temporal:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        print("\n   Validating...")
        
        with torch.no_grad():
            for frames_batch, labels in tqdm(test_loader, desc="Validation", leave=False):
                labels = labels.float().unsqueeze(1).to(device)
                
                try:
                    frames = frames_batch[0]
                    output = model(frames)
                    score = output[0]
                    
                    loss = criterion(score, labels)
                    val_loss += loss.item()
                    
                    predicted = (score > 0.5).float()
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)
                    
                except Exception as e:
                    continue
        
        epoch_val_loss = val_loss / len(test_loader) if len(test_loader) > 0 else 0
        epoch_val_acc = val_correct / val_total if val_total > 0 else 0
        
        print(f"\n   Validation Loss: {epoch_val_loss:.4f}")
        print(f"   Validation Accuracy: {epoch_val_acc*100:.2f}%")
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['temporal_contribution'].append(avg_temporal)
        
        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_epoch = epoch + 1
            
            save_path = os.path.join(save_dir, "temporal_best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': epoch_val_acc,
                'val_loss': epoch_val_loss,
                'config': config,
                'architecture': 'Temporal-Enhanced-CMAF-ExVAD'
            }, save_path)
            print(f"\n   New best model saved! (Accuracy: {epoch_val_acc*100:.2f}%)")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Training complete
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nFinal Results:")
    print(f"   Best Validation Accuracy: {best_val_acc*100:.2f}% (Epoch {best_epoch})")
    print(f"   Final Training Accuracy: {history['train_acc'][-1]*100:.2f}%")
    
    # Save history
    history_path = os.path.join(save_dir, "temporal_training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to: {history_path}")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train Temporal-Enhanced ExVAD')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--temporal_weight', type=float, default=0.3, help='Weight for temporal score')
    parser.add_argument('--no_transformer', action='store_true', help='Disable temporal transformer')
    parser.add_argument('--no_lstm', action='store_true', help='Disable temporal LSTM')
    parser.add_argument('--no_motion', action='store_true', help='Disable motion encoding')
    parser.add_argument('--freeze_base', action='store_true', help='Freeze base CMAF model')
    parser.add_argument('--data', default='data', help='Data root directory')
    parser.add_argument('--save_dir', default='models_saved', help='Save directory')
    
    args = parser.parse_args()
    
    train_temporal_model(
        epochs=args.epochs,
        learning_rate=args.lr,
        temporal_weight=args.temporal_weight,
        use_transformer=not args.no_transformer,
        use_lstm=not args.no_lstm,
        use_motion=not args.no_motion,
        freeze_base=args.freeze_base,
        data_root=args.data,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
