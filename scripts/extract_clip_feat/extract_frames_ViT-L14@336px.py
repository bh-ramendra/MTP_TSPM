import os
import torch
from PIL import Image
import numpy as np
import glob

import clip_net.clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip_net.clip.load("ViT-L/14@336px", device=device)

C = 768
patch_nums = 50


def clip_feat_extract(img):
    image = preprocess(Image.open(img)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features


def ImageClIP_Patch_feat_extract(dir_fps_path, dst_clip_path):
    os.makedirs(dst_clip_path, exist_ok=True)

    video_list = os.listdir(dir_fps_path)
    video_idx = 0
    total_nums = len(video_list)

    for video in video_list:
        video_idx += 1
        print("\n--> ", video_idx, video)

        save_file = os.path.join(dst_clip_path, video + '.npy')
        if os.path.exists(save_file):
            print(video + '.npy', "is already processed!")
            continue

        video_img_list = sorted(glob.glob(os.path.join(dir_fps_path, video, '*.jpg')))
        params_frames = len(video_img_list)

        if params_frames == 0:
            print("no frames found for:", video)
            continue

        num_samples = min(params_frames, 60)
        samples = np.round(np.linspace(0, params_frames - 1, num_samples)).astype(int)
        img_list = [video_img_list[s] for s in samples]

        img_features = torch.zeros(len(img_list), patch_nums, C)

        idx = 0
        for img_cont in img_list:
            img_idx_feat = clip_feat_extract(img_cont)
            img_features[idx] = img_idx_feat.squeeze(0).cpu()
            idx += 1

        img_features = img_features.float().cpu().numpy()
        np.save(save_file, img_features)

        print("Process: ", video_idx, " / ", total_nums,
              " ----- video id: ", video,
              " ----- save shape: ", img_features.shape)


def ImageClIP_feat_extract(dir_fps_path, dst_clip_path):
    os.makedirs(dst_clip_path, exist_ok=True)

    video_list = os.listdir(dir_fps_path)

    video_idx = 0
    total_nums = len(video_list)

    for video in video_list:
        video_idx += 1
        print("\n--> ", video_idx, video)

        save_file = os.path.join(dst_clip_path, video + '.npy')
        if os.path.exists(save_file):
            print(video + '.npy', "is already processed!")
            continue

        video_img_list = sorted(glob.glob(os.path.join(dir_fps_path, video, '*.jpg')))

        params_frames = len(video_img_list)
        if params_frames == 0:
            print("no frames found for:", video)
            continue

        num_samples = min(params_frames, 60)
        samples = np.round(np.linspace(0, params_frames - 1, num_samples)).astype(int)
        img_list = [video_img_list[s] for s in samples]

        img_features = torch.zeros(len(img_list), C)

        idx = 0
        for img_cont in img_list:
            img_idx_feat = clip_feat_extract(img_cont)
            img_features[idx] = img_idx_feat.squeeze(0).cpu()
            idx += 1

        img_features = img_features.float().cpu().numpy()
        np.save(save_file, img_features)

        print("Process: ", video_idx, " / ", total_nums,
              " ----- video id: ", video,
              " ----- save shape: ", img_features.shape)


if __name__ == "__main__":
    dir_fps_path = '/DATA/ramendra_2511ai39/MUSIC-AVQA-Dataset/avqa-frames-1fps'
    dst_clip_path = '/DATA/ramendra_2511ai39/TSPM/clip_feats/frame_1fps_ViT-L14@336px'
    ImageClIP_feat_extract(dir_fps_path, dst_clip_path)