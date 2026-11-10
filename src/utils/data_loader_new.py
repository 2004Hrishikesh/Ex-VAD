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
        
        # Class mapping - all anomaly classes map to 1
        # Handle both lowercase and capitalized names
        normal_names = ['normal', 'Normal', 'NormalVideos']
        anomaly_classes = ['abuse', 'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 
                          'explosion', 'Explosion', 'Fighting', 'RoadAccidents', 'Robbery', 
                          'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']
        
        self.class_to_idx = {}
        for normal_name in normal_names:
            self.class_to_idx[normal_name] = 0
        for anomaly_class in anomaly_classes:
            self.class_to_idx[anomaly_class] = 1
        
        self._load_videos()
        
        print(f"Loaded {len(self.video_paths)} videos for {split}")
        print(f"Normal: {self.labels.count(0)}, Anomaly: {self.labels.count(1)}")
    
    def _load_videos(self):
        # Try both path structures: data/videos/train and data/Train
        split_path = os.path.join(self.data_root, 'videos', self.split)
        if not os.path.exists(split_path):
            # Try capitalized version
            split_path = os.path.join(self.data_root, self.split.capitalize())
            if not os.path.exists(split_path):
                print(f"Warning: Path {split_path} does not exist")
                return
            
        valid_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv')
        
        # Get all class folders dynamically
        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if not os.path.isdir(class_path):
                continue
            
            # Skip Labels folder
            if class_name == 'Labels':
                continue
            
            # Map class name to label
            if class_name not in self.class_to_idx:
                print(f"Warning: Unknown class '{class_name}', treating as anomaly")
                label = 1  # Treat unknown as anomaly
            else:
                label = self.class_to_idx[class_name]
            
            try:
                entries = os.listdir(class_path)
                
                # Check if folder contains video files or frame sequences
                video_files = [e for e in entries if e.lower().endswith(valid_extensions)]
                frame_files = [e for e in entries if e.lower().endswith('.png')]
                
                if video_files:
                    # Process video files
                    for entry in video_files:
                        entry_path = os.path.join(class_path, entry)
                        if os.path.getsize(entry_path) > 1000:  # At least 1KB
                            self.video_paths.append(entry_path)
                            self.labels.append(label)
                            self.class_names.append(class_name)
                
                elif frame_files:
                    # Process frame sequences - group by video name
                    # Extract video names efficiently
                    video_groups = {}
                    for frame_file in frame_files:
                        # Extract video name (e.g., "Robbery001_x264_0.png" -> "Robbery001_x264")
                        # Split by underscore and remove the last numeric part and extension
                        parts = frame_file.rsplit('_', 1)
                        if len(parts) == 2 and parts[1].replace('.png', '').isdigit():
                            video_name = parts[0]
                        else:
                            video_name = frame_file.split('.')[0]
                        
                        if video_name not in video_groups:
                            video_groups[video_name] = 0
                        video_groups[video_name] += 1
                    
                    # Add each video group as a single entry
                    for video_name, frame_count in video_groups.items():
                        if frame_count >= self.num_frames:  # Must have enough frames
                            # Store the folder path and video name
                            video_path = os.path.join(class_path, video_name)
                            self.video_paths.append(video_path)
                            self.labels.append(label)
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
            # Check if it's a video file or frame sequence
            if os.path.isfile(video_path):
                # It's a video file
                frames = sample_frames(video_path, self.num_frames, self.device)
            else:
                # It's a frame sequence (directory indicator)
                frames = self._load_frame_sequence(video_path)
            
            return frames, label, class_name, video_path
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            # Return dummy data
            dummy_frames = [Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)) 
                          for _ in range(self.num_frames)]
            return dummy_frames, 0, 'normal', video_path
    
    def _load_frame_sequence(self, video_path):
        """Load frames from a PNG sequence"""
        # video_path is like "data/Train/Robbery/Robbery001_x264"
        # Find all matching frame files
        parent_dir = os.path.dirname(video_path)
        video_name = os.path.basename(video_path)
        
        # Get all frame files for this video
        all_files = os.listdir(parent_dir)
        frame_files = sorted([f for f in all_files if f.startswith(video_name) and f.endswith('.png')])
        
        if not frame_files:
            raise ValueError(f"No frame files found for {video_name}")
        
        # Sample frames uniformly
        total_frames = len(frame_files)
        if total_frames >= self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            selected_files = [frame_files[i] for i in indices]
        else:
            # Repeat frames if not enough
            selected_files = frame_files + [frame_files[-1]] * (self.num_frames - total_frames)
        
        # Load frames
        frames = []
        for frame_file in selected_files:
            frame_path = os.path.join(parent_dir, frame_file)
            frame = Image.open(frame_path).convert('RGB')
            # Resize to 224x224
            frame = frame.resize((224, 224), Image.Resampling.LANCZOS)
            frames.append(frame)
        
        return frames

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
