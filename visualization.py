import cv2
import os
from Demos.BackupSeek_streamheaders import parse_stream_header
from sympy.codegen import Print

from utils import normalize
import numpy as np


def visualizer(images, anomaly_map, img_size, save_path, counter,text_probs,path):
    image = images[0]  # 转为 NumPy 数组，大小为 [3, 640, 640]
    anomaly = anomaly_map[0] # 转为 NumPy 数组，大小为 [640, 640]
    image = np.transpose(image, (1, 2, 0))  #  [3, 518, 518] -> [518, 518, 3]

    vis = cv2.resize(image, (img_size, img_size))  # Resize to target size
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)  # Convert to RGB format

    mask = normalize(anomaly)
    vis = apply_ad_scoremap(vis, mask)
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    filename = f"{path}.png"
    save_vis = os.path.join(save_path, 'zero-shot')
    if not os.path.exists(save_vis):
        os.makedirs(save_vis)
    cv2.imwrite(os.path.join(save_vis, filename), vis)
    counter += 1
    return counter


def apply_ad_scoremap(image, scoremap, alpha=0.1):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

