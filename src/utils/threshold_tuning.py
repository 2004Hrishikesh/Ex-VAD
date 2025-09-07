import torch
import numpy as np
from tqdm import tqdm

def find_optimal_threshold(model, val_loader, device, thresholds=None):
    """Find the optimal threshold that maximizes F1 score on validation set"""
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 50)
    
    model.eval()
    all_scores = []
    all_labels = []
    
    print("Collecting validation predictions...")
    with torch.no_grad():
        for frames_batch, labels, _, _ in tqdm(val_loader):
            frames = frames_batch[0]
            label = labels[0].float()
            
            if len(frames) == 0:
                continue
                
            try:
                anomaly_score, _, _ = model(frames)
                if isinstance(anomaly_score, torch.Tensor):
                    score = torch.sigmoid(anomaly_score).cpu().item()
                else:
                    score = 0.5
                
                all_scores.append(score)
                all_labels.append(label.item())
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    best_threshold = 0.5
    best_f1 = 0
    best_metrics = None
    
    print("\nTuning threshold...")
    for threshold in tqdm(thresholds):
        predictions = (all_scores > threshold).astype(int)
        
        # Calculate metrics
        tp = np.sum((predictions == 1) & (all_labels == 1))
        fp = np.sum((predictions == 1) & (all_labels == 0))
        fn = np.sum((predictions == 0) & (all_labels == 1))
        tn = np.sum((predictions == 0) & (all_labels == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(all_labels)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'threshold': threshold,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy
            }
    
    print(f"\nOptimal threshold: {best_threshold:.3f}")
    print(f"F1 Score: {best_metrics['f1']:.3f}")
    print(f"Precision: {best_metrics['precision']:.3f}")
    print(f"Recall: {best_metrics['recall']:.3f}")
    print(f"Accuracy: {best_metrics['accuracy']:.3f}")
    
    return best_threshold, best_metrics
