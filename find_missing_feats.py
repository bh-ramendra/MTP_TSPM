from pathlib import Path
import json

root = Path(".")
split_dir = root / "dataset" / "split_video_id"
feat_dir = Path("/DATA/ramendra_2511ai39/TSPM/clip_feats/frame_1fps_ViT-L14@336px")

video_ids = set()

for p in split_dir.glob("music_avqa_*.json"):
    data = json.loads(p.read_text())
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                if "video_id" in item:
                    video_ids.add(str(item["video_id"]))
                elif "video" in item:
                    video_ids.add(str(item["video"]))
                elif "vid" in item:
                    video_ids.add(str(item["vid"]))
                elif "id" in item:
                    video_ids.add(str(item["id"]))
            else:
                video_ids.add(str(item))
    elif isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                for item in v:
                    video_ids.add(str(item))
            else:
                video_ids.add(str(v))

missing = sorted([vid for vid in video_ids if not (feat_dir / f"{vid}.npy").exists()])

print("total_ids =", len(video_ids))
print("missing =", len(missing))

Path("missing_ids.txt").write_text("\n".join(missing) + "\n")