import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from .video_utils import sample_frames

class VideoAnomalyDataset(Dataset):
    def __init__(self, data_root, split='train', num_frames=16, device='cuda'):
        self.data_root = data_root
        self.split = split
        self.num_frames = num_frames
        self.device = device
        self.video_paths = []
        self.labels = []
        self.class_names = []
        self.failed_loads = []
        
        # Class mapping
        self.class_to_idx = {
            'normal': 0,
            'abuse': 1,
            'explosion': 1
        }
        
        # Data augmentation for training
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0))
            ])
        else:
            self.transform = None
        
        self._load_videos()
        
        # Print dataset statistics
        normal_count = self.labels.count(0)
        anomaly_count = self.labels.count(1)
        total_count = len(self.video_paths)
        
        print(f"\nDataset Statistics for {split}:")
        print(f"Total videos: {total_count}")
        print(f"Normal videos: {normal_count} ({100 * normal_count / total_count:.1f}%)")
        print(f"Anomaly videos: {anomaly_count} ({100 * anomaly_count / total_count:.1f}%)")
        if self.failed_loads:
            print(f"Failed to load: {len(self.failed_loads)} videos")
        
        # Calculate class weights for balanced training
        self.class_weights = {
            0: 1.0 / (normal_count if normal_count > 0 else 1),
            1: 1.0 / (anomaly_count if anomaly_count > 0 else 1)
        }
    
    def _load_videos(self):
        """Load video paths and validate them"""
        split_path = os.path.join(self.data_root, 'videos', self.split)
        valid_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv')
        min_size = 1000  # Minimum file size in bytes
        
        for class_name in ['normal', 'abuse', 'explosion']:
            class_path = os.path.join(split_path, class_name)
            if not os.path.exists(class_path):
                print(f"Warning: Class path {class_path} does not exist")
                continue
            
            # Walk through directory tree
            for root, _, files in os.walk(class_path):
                for file in files:
                    if file.lower().endswith(valid_extensions):
                        video_path = os.path.join(root, file)
                        try:
                            # Validate video file
                            if os.path.getsize(video_path) < min_size:
                                self.failed_loads.append((video_path, "File too small"))
                                continue
                                
                            # Try to read first frame to validate video
                            test_frames = sample_frames(video_path, 1, self.device)
                            if test_frames is None or len(test_frames) == 0:
                                self.failed_loads.append((video_path, "Cannot read frames"))
                                continue
                            
                            # Add valid video
                            self.video_paths.append(video_path)
                            self.labels.append(self.class_to_idx[class_name])
                            self.class_names.append(class_name)
                            
                        except Exception as e:
                            self.failed_loads.append((video_path, str(e)))
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        """Get video frames with augmentation if in training mode"""
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        class_name = self.class_names[idx]
        weight = self.class_weights[label]
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Load frames
                frames = sample_frames(video_path, self.num_frames, self.device)
                
                # Apply data augmentation in training mode
                if self.transform is not None:
                    augmented_frames = []
                    for frame in frames:
                        aug_frame = self.transform(frame)
                        augmented_frames.append(aug_frame)
                    frames = augmented_frames
                
                return frames, label, class_name, video_path, weight
            
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Error loading {video_path} after {max_retries} attempts: {e}")
                    # Return dummy data with high weight to ensure it's noticed
                    dummy_frames = [Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)) 
                                 for _ in range(self.num_frames)]
                    return dummy_frames, 0, 'normal', video_path, 1.0

def custom_collate_fn(batch):
    """Custom collate function for video data with class weights"""
    frames_list = []
    labels_list = []
    class_names_list = []
    video_paths_list = []
    weights_list = []
    
    for frames, label, class_name, video_path, weight in batch:
        frames_list.append(frames)
        labels_list.append(label)
        class_names_list.append(class_name)
        video_paths_list.append(video_path)
        weights_list.append(weight)
    
    return (
        frames_list, 
        torch.tensor(labels_list), 
        class_names_list, 
        video_paths_list,
        torch.tensor(weights_list)
    )

def create_dataloaders(data_root, batch_size=1, num_frames=16, device='cuda', num_workers=0):
    """Create train and test dataloaders with optimal settings"""
    if not os.path.exists(data_root):
        raise ValueError(f"Data root path {data_root} does not exist")
    
    # Create datasets
    train_dataset = VideoAnomalyDataset(data_root, 'train', num_frames, device)
    test_dataset = VideoAnomalyDataset(data_root, 'test', num_frames, device)
    
    # DataLoader settings
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': device == 'cuda',
        'persistent_workers': num_workers > 0,
        'collate_fn': custom_collate_fn
    }
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,  # Shuffle training data
        **loader_kwargs
    )
    
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,  # Don't shuffle test data
        **loader_kwargs
    )
    
    return train_loader, test_loader
