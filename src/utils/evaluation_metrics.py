import numpy as np
import torch
from sklearn.metrics import (
    roc_curve, 
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_metrics(y_true, y_pred_probs, threshold=0.5):
    """
    Calculate comprehensive metrics for anomaly detection
    
    Args:
        y_true: Ground truth labels (0: normal, 1: anomaly)
        y_pred_probs: Predicted probabilities for anomaly class
        threshold: Classification threshold
    
    Returns:
        dict: Dictionary containing all metrics
    """
    # Ensure numpy arrays
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    
    # Binary predictions using threshold
    y_pred = (y_pred_probs >= threshold).astype(int)
    
    # Basic confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = f1_score(y_true, y_pred)
    
    # ROC and AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    auc_roc = roc_auc_score(y_true, y_pred_probs)
    
    # Precision-Recall curve
    precisions, recalls, _ = precision_recall_curve(y_true, y_pred_probs)
    auc_pr = average_precision_score(y_true, y_pred_probs)
    
    # False Alarm Rate (FAR) and Missing Rate (MR)
    far = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Alarm Rate
    mr = fn / (fn + tp) if (fn + tp) > 0 else 0   # Missing Rate
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'far': far,
        'mr': mr,
        'confusion_matrix': {
            'tn': tn, 'fp': fp,
            'fn': fn, 'tp': tp
        },
        'roc_curve': {
            'fpr': fpr,
            'tpr': tpr
        },
        'pr_curve': {
            'precisions': precisions,
            'recalls': recalls
        }
    }

def plot_metrics(metrics, save_path=None):
    """
    Plot evaluation metrics including ROC curve, PR curve, and confusion matrix
    
    Args:
        metrics: Dictionary of metrics from calculate_metrics()
        save_path: Path to save the plots (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle('Anomaly Detection Evaluation Metrics', fontsize=16)
    
    # ROC Curve
    ax = axes[0, 0]
    ax.plot(metrics['roc_curve']['fpr'], 
            metrics['roc_curve']['tpr'], 
            label=f'AUC = {metrics["auc_roc"]:.3f}')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True)
    
    # Precision-Recall Curve
    ax = axes[0, 1]
    ax.plot(metrics['pr_curve']['recalls'], 
            metrics['pr_curve']['precisions'],
            label=f'AUC = {metrics["auc_pr"]:.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    ax.grid(True)
    
    # Confusion Matrix
    ax = axes[1, 0]
    cm = np.array([[metrics['confusion_matrix']['tn'], metrics['confusion_matrix']['fp']],
                  [metrics['confusion_matrix']['fn'], metrics['confusion_matrix']['tp']]])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    
    # Metrics Summary
    ax = axes[1, 1]
    ax.axis('off')
    summary = f"""
    Summary Metrics:
    
    Accuracy: {metrics['accuracy']:.3f}
    Precision: {metrics['precision']:.3f}
    Recall: {metrics['recall']:.3f}
    F1 Score: {metrics['f1_score']:.3f}
    AUC-ROC: {metrics['auc_roc']:.3f}
    AUC-PR: {metrics['auc_pr']:.3f}
    
    False Alarm Rate: {metrics['far']:.3f}
    Missing Rate: {metrics['mr']:.3f}
    """
    ax.text(0.1, 0.5, summary, fontsize=12, va='center')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def evaluate_model(model, test_loader, device):
    """
    Evaluate model performance using comprehensive metrics
    
    Args:
        model: The trained model
        test_loader: DataLoader for test data
        device: Device to run evaluation on
    
    Returns:
        dict: Dictionary containing all metrics
    """
    model.eval()
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for frames_batch, labels, _, _ in test_loader:
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
                print(f"Error in evaluation: {e}")
                continue
    
    metrics = calculate_metrics(all_labels, all_scores)
    return metrics
