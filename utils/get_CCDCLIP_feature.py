import argparse
import os
import csv
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision.io import read_video
from TrafficCLIP_lib import load
from utils import get_transform

def extract_features(video_path, model, preprocess, device):
    try:
        video, _, _ = read_video(video_path, pts_unit='sec')
        frames = video[:50]  # CCD 每个视频 50 帧
        features = []
        for frame in frames:
            img = Image.fromarray(frame.numpy())
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.encode_image(img_tensor)
            features.append(feat.squeeze(0).cpu())
        return torch.stack(features).numpy()
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None

def get_CCD_feature(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load("ViT-L/14@336px", device=device, jit=False)
    model = model.eval()
    preprocess, _ = get_transform(args)

    base_crash_dir = "D:/CCDdataset/Crash-1500"
    base_normal_dir = "D:/CCDdataset/Normal"
    output_feature_dir = "D:/CCD_features"
    os.makedirs(output_feature_dir, exist_ok=True)

    splits = {
        "train": open("CCD_feature_train.csv", "w", newline=''),
        "test": open("CCD_feature_test.csv", "w", newline='')
    }
    writers = {k: csv.writer(f) for k, f in splits.items()}

    for split in ["train", "test"]:
        txt_path = f"D:/CCDdataset/{split}.txt"
        with open(txt_path, "r") as f:
            lines = f.readlines()

        for line in tqdm(lines, desc=f"Processing {split}"):
            rel_path, label = line.strip().split()
            label = int(label)
            category, filename = rel_path.split("/")
            video_id = filename.replace(".npz", "")

            if category == "positive":
                video_path = os.path.join(base_crash_dir, f"{video_id}.mp4")
            else:
                video_path = os.path.join(base_normal_dir, f"{video_id}.mp4")

            features = extract_features(video_path, model, preprocess, device)
            if features is not None:
                npy_name = f"{split}_{category}_{video_id}.npy"
                npy_path = os.path.join(output_feature_dir, npy_name)
                np.save(npy_path, features)
                label_str = "accident" if label == 1 else "no accident"
                writers[split].writerow([npy_path, label_str])

    for f in splits.values():
        f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("CCD Feature Extractor")
    parser.add_argument("--image_size", type=int, default=640)
    args = parser.parse_args()
    get_CCD_feature(args)
