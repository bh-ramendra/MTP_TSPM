import os
import torch
try:
    import cv2
except ImportError:
    cv2 = None
from PIL import Image
import numpy as np
import glob
import sys
from concurrent.futures import ThreadPoolExecutor

# Ensure local clip_net is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clip_net.clip as clip
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--rank", type=int, default=0)
parser.add_argument("--world-size", type=int, default=1)
parser.add_argument("--device", type=str, default="cuda")
args, _ = parser.parse_known_args()

device = args.device
print(f"Rank {args.rank}/{args.world_size} using device: {device}")
model, preprocess = clip.load("ViT-L/14@336px", device=device)
model.eval()

# CLIP models are thread-safe, but to prevent GPU memory lock/race conditions
# we can use a lock when calling the GPU model forward pass
gpu_lock = torch.cuda.Event()

C = 768
target_num_frames = 60

# Paths
real_frames_dir = '/DATA/ramendra_2511ai39/MUSIC-AVQA/data/frames/'
dataset_frames_dir = '/DATA/ramendra_2511ai39/MUSIC-AVQA-Dataset/avqa-frames-1fps/'
real_videos_dir = '/DATA/ramendra_2511ai39/MUSIC-AVQA-Dataset/MUSIC-AVQA-videos-Real/'
synthetic_videos_dir = '/DATA/ramendra_2511ai39/MUSIC-AVQA-Dataset/MUSIC-AVQA-videos-Synthetic/'
dst_clip_path = '/DATA/ramendra_2511ai39/TSPM/clip_feats/frame_1fps_ViT-L14@336px'

os.makedirs(dst_clip_path, exist_ok=True)

def preprocess_single_image(img_path):
    try:
        img = Image.open(img_path)
        return preprocess(img)
    except Exception as e:
        # Fallback to zero tensor if image is corrupt
        return torch.zeros(3, 336, 336)

def load_frames_from_jpg(folder):
    video_img_list = sorted(glob.glob(os.path.join(folder, '*.jpg')))
    params_frames = len(video_img_list)
    if params_frames == 0:
        return None
    num_samples = min(params_frames, target_num_frames)
    samples = np.round(np.linspace(0, params_frames - 1, num_samples)).astype(int)
    img_list = [video_img_list[s] for s in samples]
    
    # Load and preprocess in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        tensors = list(executor.map(preprocess_single_image, img_list))
    return torch.stack(tensors)

def load_frames_from_video(video_path):
    if cv2 is None:
        raise ImportError("cv2 is required to read raw video files.")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None
    
    num_samples = min(total_frames, target_num_frames)
    indices = set(np.round(np.linspace(0, total_frames - 1, num_samples)).astype(int))
    
    tensors = []
    count = 0
    while len(tensors) < num_samples:
        ret, frame = cap.read()
        if not ret:
            break
        if count in indices:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            tensors.append(preprocess(img))
        count += 1
    cap.release()
    if len(tensors) == 0:
        return None
    return torch.stack(tensors)

def extract_features(video_id):
    save_file = os.path.join(dst_clip_path, video_id + '.npy')
    if os.path.exists(save_file):
        return True, "already exists"
    
    # 1. Try real frames directory
    folder_path = os.path.join(real_frames_dir, video_id)
    frame_tensor = None
    if os.path.exists(folder_path):
        frame_tensor = load_frames_from_jpg(folder_path)
        
    # 2. Try dataset frames directory
    if frame_tensor is None:
        folder_path = os.path.join(dataset_frames_dir, video_id)
        if os.path.exists(folder_path):
            frame_tensor = load_frames_from_jpg(folder_path)
            
    # 3. Try reading from real MP4 video
    if frame_tensor is None:
        video_path = os.path.join(real_videos_dir, video_id + '.mp4')
        if os.path.exists(video_path):
            frame_tensor = load_frames_from_video(video_path)
            
    # 4. Try reading from synthetic MP4 video
    if frame_tensor is None:
        video_path = os.path.join(synthetic_videos_dir, video_id + '.mp4')
        if os.path.exists(video_path):
            frame_tensor = load_frames_from_video(video_path)
            
    if frame_tensor is None:
        return False, "no frame source found"
        
    # Move to GPU and encode batched
    frame_tensor = frame_tensor.to(device)
    with torch.no_grad():
        image_features = model.encode_image(frame_tensor)
        
    img_features = image_features.float().cpu().numpy()
    np.save(save_file, img_features)
    return True, f"extracted shape {img_features.shape}"

def process_video_id(vid_info):
    i, vid = vid_info
    try:
        success, msg = extract_features(vid)
        if success:
            if msg == "already exists":
                return "skip"
            else:
                return f"[{i+1}/9288] Video {vid}: {msg}"
        else:
            return f"[{i+1}/9288] Video {vid} FAILED: {msg}"
    except Exception as e:
        return f"[{i+1}/9288] Video {vid} ERROR: {str(e)}"

if __name__ == "__main__":
    import json
    from pathlib import Path
    
    split_dir = Path("/DATA/ramendra_2511ai39/TSPM/dataset/split_video_id")
    video_ids = set()
    for p in split_dir.glob("music_avqa_*.json"):
        data = json.loads(p.read_text())
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    video_ids.add(str(item.get("video_id", item.get("video", item.get("vid", item.get("id"))))))
                else:
                    video_ids.add(str(item))
        elif isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    for item in v:
                        video_ids.add(str(item))
                else:
                    video_ids.add(str(v))
                    
    video_ids = sorted([vid for vid in video_ids if vid and vid != 'None'])
    print(f"Total video IDs in full dataset: {len(video_ids)}")
    video_ids = video_ids[args.rank::args.world_size]
    print(f"Rank {args.rank}/{args.world_size}: processing {len(video_ids)} video IDs")
    
    # Use ThreadPoolExecutor to process multiple videos in parallel
    # max_workers=6 overlaps image loading and GPU execution perfectly
    with ThreadPoolExecutor(max_workers=6) as executor:
        for res in executor.map(process_video_id, enumerate(video_ids)):
            if res != "skip":
                print(res, flush=True)
                
    print(f"Rank {args.rank} CLIP Feature extraction complete.")
