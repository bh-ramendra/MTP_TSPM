# TSPM (Key Semantic-Aware Cues for Audio-Visual QA)

Official repository for **"Boosting Audio Visual Question Answering via Key Semantic-Aware Cues"** (ACM MM 2024).

- **ArXiv Paper**: [Boosting Audio Visual Question Answering via Key Semantic-Aware Cues](https://arxiv.org/abs/2407.20693)
- **Authors**: [Guangyao Li](https://ayameyao.github.io/), [Henghui Du](), [Di Hu](https://dtaoo.github.io/index.html)

---

## 🛠️ Requirements & Installation

### Environment Setup
We recommend setting up a virtual environment or Conda environment with Python 3.6+:

```bash
# Example conda environment creation
conda create -n avqa python=3.8 -y
conda activate avqa
```

Install the required packages:
* PyTorch (>= 1.6.0)
* tensorboardX
* ffmpeg-python
* opencv-python
* numpy
* pillow

---

## 🚀 Getting Started

### 1. Clone this Repository
```bash
git clone git@github.com:bh-ramendra/MTP_TSPM.git
cd MTP_TSPM
```

### 2. Download the Datasets
Download the datasets using the official links:
* **MUSIC-AVQA**: [MUSIC-AVQA Homepage](https://gewu-lab.github.io/MUSIC-AVQA/)
* **AVQA**: [AVQA Tsinghua Homepage](http://mn.cs.tsinghua.edu.cn/avqa/)

Organize your raw video files and extracted frame folders in your data directory.

---

## ⚡ Feature Extraction Pipeline

We provide both original extraction scripts and **newly optimized fast scripts** that leverage multi-threading and multiprocessing.

### A. Fast & Parallel Frame Feature Extraction (New)
* **`scripts/extract_clip_feat/extract_frames_fast.py`**:
  Uses `ThreadPoolExecutor` (default 6 overlapping workers) to load and preprocess images in parallel while using GPU locks to prevent race conditions during CLIP encoding. This script falls back across different paths automatically:
  1. Real frames JPG folder
  2. Dataset frames JPG folder
  3. Direct frame extraction from Real `.mp4` video files
  4. Direct frame extraction from Synthetic `.mp4` video files

  Run using:
  ```bash
  python scripts/extract_clip_feat/extract_frames_fast.py --device cuda
  ```

* **`scripts/extract_clip_feat/extract_synthetic_jpgs.py`**:
  Uses a parallel multiprocessing pool (24 workers) to extract 1fps frames from raw synthetic videos, saving them as sequential `.jpg` images for faster consumption.
  ```bash
  python scripts/extract_clip_feat/extract_synthetic_jpgs.py
  ```

* **`find_missing_feats.py`**:
  Scans all splits (train/val/test) json files, identifies all required video IDs, checks if their corresponding `.npy` frame feature files exist in the feature directory, and exports any missing IDs to `missing_ids.txt`.
  ```bash
  python find_missing_feats.py
  ```

### B. Standard Feature Extraction
If you prefer using the original sequential extraction scripts, run them from the extraction directory:
```bash
cd scripts/extract_clip_feat/
python extract_qst_clip_feat.py
python extract_qaPrompt_ViT-L14@336px.py
python extract_token-level_feat.py
python extract_frames_ViT-L14@336px.py
```

---

## 🏋️ Training & Evaluation

### Configuration
You can customize directories, paths, GPUs, batch sizes, and model hyperparameters in:
* [`configs/arguments.py`](file:///DATA/ramendra_2511ai39/TSPM/configs/arguments.py)

### Training
Use the helper script to run training with preset optimal hyperparameters:
```bash
bash run_train.sh
```

Or run the Python command directly:
```bash
python -u main_train.py \
    --Temp_Selection \
    --top_k 10 \
    --Spatio_Perception \
    --batch-size 8 \
    --epochs 30 \
    --lr 1e-4 \
    --num_workers 4 \
    --gpu 1 \
    --checkpoint TSPM_local \
    --model_save_dir models
```

### Testing & Evaluation
Test your trained checkpoints using:
```bash
python -u main_test.py \
    --Temp_Selection \
    --top_k 10 \
    --Spatio_Perception \
    --batch-size 1 \
    --gpu 1 \
    --checkpoint TSPM_local \
    --model_save_dir models \
    --result_dir results
```

---

## 📝 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{li2024boosting,
  title={Boosting Audio Visual Question Answering via Key Semantic-Aware Cues},
  author={Li, Guangyao and Du, Henghui and Hu, Di},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia (ACM MM)},
  year={2024}
}
```

---

## 🤝 Acknowledgement
This research was supported by the Public Computing Cloud, Renmin University of China.
