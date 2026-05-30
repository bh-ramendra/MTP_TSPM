import os
import cv2
import numpy as np
import glob
from multiprocessing import Pool

synthetic_videos_dir = '/DATA/ramendra_2511ai39/MUSIC-AVQA-Dataset/MUSIC-AVQA-videos-Synthetic/'
dst_frames_dir = '/DATA/ramendra_2511ai39/MUSIC-AVQA-Dataset/avqa-frames-1fps/'
target_num_frames = 60

def extract_single_video(video_path):
    video_name = os.path.basename(video_path).replace('.mp4', '')
    save_folder = os.path.join(dst_frames_dir, video_name)
    
    # Check if folder already exists and has frames
    if os.path.exists(save_folder) and len(glob.glob(os.path.join(save_folder, '*.jpg'))) >= target_num_frames:
        return True, "already exists"
        
    os.makedirs(save_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "error opening video"
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return False, "invalid total frames"
        
    num_samples = min(total_frames, target_num_frames)
    indices = np.round(np.linspace(0, total_frames - 1, num_samples)).astype(int)
    
    # Read frames sequentially for speed
    count = 0
    sample_idx = 0
    
    while sample_idx < len(indices):
        ret, frame = cap.read()
        if not ret:
            break
        if count == indices[sample_idx]:
            filename = os.path.join(save_folder, f"{sample_idx:06d}.jpg")
            cv2.imwrite(filename, frame)
            sample_idx += 1
        count += 1
        
    cap.release()
    return True, "success"

if __name__ == "__main__":
    video_files = sorted(glob.glob(os.path.join(synthetic_videos_dir, '*.mp4')))
    print(f"Found {len(video_files)} synthetic videos. Starting parallel extraction with 24 workers...")
    
    # Parallel processing
    with Pool(24) as p:
        results = p.map(extract_single_video, video_files)
        
    success = sum(1 for r in results if r[0])
    skipped = sum(1 for r in results if r[0] and r[1] == "already exists")
    failed = len(video_files) - success
    
    print(f"Finished extraction. Success: {success - skipped}, Skipped: {skipped}, Failed: {failed}")
