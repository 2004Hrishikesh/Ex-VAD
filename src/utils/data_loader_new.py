import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
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
        
        # Class mapping
        self.class_to_idx = {
            'normal': 0,
            'abuse': 1,
            'explosion': 1
        }
        
        self._load_videos()
        
        print(f"Loaded {len(self.video_paths)} videos for {split}")
        print(f"Normal: {self.labels.count(0)}, Anomaly: {self.labels.count(1)}")
    
    def _load_videos(self):
        split_path = os.path.join(self.data_root, 'videos', self.split)
        if not os.path.exists(split_path):
            print(f"Warning: Path {split_path} does not exist")
            return
            
        valid_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv')
        
        for class_name in ['normal', 'abuse', 'explosion']:
            class_path = os.path.join(split_path, class_name)
            if not os.path.exists(class_path):
                print(f"Warning: Class path {class_path} does not exist")
                continue
            
            try:
                for entry in os.listdir(class_path):
                    entry_path = os.path.join(class_path, entry)
                    
                    # Skip non-video files
                    if not entry.lower().endswith(valid_extensions):
                        continue
                        
                    # Check file size
                    if os.path.getsize(entry_path) > 1000:  # At least 1KB
                        self.video_paths.append(entry_path)
                        self.labels.append(self.class_to_idx[class_name])
                        self.class_names.append(class_name)
            except Exception as e:
                print(f"Error loading from {class_path}: {e}")
                continue
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        class_name = self.class_names[idx]
        
        try:
            frames = sample_frames(video_path, self.num_frames, self.device)
            return frames, label, class_name, video_path
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            # Return dummy data
            dummy_frames = [Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)) 
                          for _ in range(self.num_frames)]
            return dummy_frames, 0, 'normal', video_path

def custom_collate_fn(batch):
    """Custom collate function for video data"""
    frames_list = []
    labels_list = []
    class_names_list = []
    video_paths_list = []
    
    for frames, label, class_name, video_path in batch:
        frames_list.append(frames)
        labels_list.append(label)
        class_names_list.append(class_name)
        video_paths_list.append(video_path)
    
    return frames_list, torch.tensor(labels_list), class_names_list, video_paths_list

def create_dataloaders(data_root, batch_size=1, num_frames=16, device='cuda', num_workers=0):
    """Create train and test dataloaders with optimal settings"""
    if not os.path.exists(data_root):
        raise ValueError(f"Data root path {data_root} does not exist")
        
    train_dataset = VideoAnomalyDataset(data_root, 'train', num_frames, device)
    test_dataset = VideoAnomalyDataset(data_root, 'test', num_frames, device)
    
    loader_args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': device == 'cuda',
        'persistent_workers': num_workers > 0,
        'collate_fn': custom_collate_fn
    }
    
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)
    
    return train_loader, test_loader
