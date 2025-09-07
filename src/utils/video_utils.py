import cv2
import numpy as np
from PIL import Image
import torch

def sample_frames(video_path, num_frames=16, device='cuda'):
    """
    Sample frames from video with GPU optimization
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"No frames in video: {video_path}")
    
    # Sample frame indices
    if total_frames <= num_frames:
        indices = list(range(total_frames))
        indices.extend([total_frames-1] * (num_frames - total_frames))
    else:
        indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        
        if ret and frame is not None:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            frame = Image.fromarray(frame)
            frames.append(frame)
        elif frames:
            # If frame reading fails, repeat last frame
            frames.append(frames[-1])
        else:
            # If first frame fails, create a black frame
            black_frame = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            frames.append(black_frame)
    
    cap.release()
    
    # Ensure we have exactly num_frames
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)))
    
    return frames[:num_frames]

def frames_to_tensor(frames, device='cuda'):
    """Convert PIL frames to tensor batch"""
    # Convert frames to numpy arrays and stack
    frame_arrays = []
    for frame in frames:
        # Resize to 224x224 if needed
        if frame.size != (224, 224):
            frame = frame.resize((224, 224))
        frame_array = np.array(frame).astype(np.float32) / 255.0
        frame_arrays.append(frame_array)
    
    # Stack and convert to tensor
    video_tensor = np.stack(frame_arrays, axis=0)  # Shape: (T, H, W, C)
    video_tensor = torch.from_numpy(video_tensor).permute(0, 3, 1, 2)  # Shape: (T, C, H, W)
    
    return video_tensor.to(device)
