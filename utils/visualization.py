import cv2
import os

from Demos.BackupSeek_streamheaders import parse_stream_header
from sympy.codegen import Print

from utils import normalize
import numpy as np

def visualizer(images, anomaly_map, img_size, save_path, counter,text_probs,path):
    # 假设 images 的大小是 [1, 3, 518, 518]，anomaly_map 的大小是 [1, 518, 518]
    # 取出对应的图像和异常图
    image = images[0]  # 转为 NumPy 数组，大小为 [3, 640, 640]
    #print(image.shape)
    anomaly = anomaly_map[0] # 转为 NumPy 数组，大小为 [640, 640]
    #print(anomaly.shape)

    # 将图像从 [3, 518, 518] 转换为 [518, 518, 3]，调整为 HWC 格式
    image = np.transpose(image, (1, 2, 0))  # 从 [3, 518, 518] -> [518, 518, 3]

    # 确保图像大小符合要求
    vis = cv2.resize(image, (img_size, img_size))  # Resize to target size
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)  # Convert to RGB format

    # 归一化异常图
    mask = normalize(anomaly)  # 假设 normalize 会将 anomaly 归一化到 [0, 1]

    # 将异常图应用到图像上
    vis = apply_ad_scoremap(vis, mask)

    # 将图像恢复为 BGR 格式
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    #print(f"vis 的维度: {vis.shape}") (640,640,3)

    # 使用计数器生成文件名
    filename = f"{path}.png"  # 文件名为 test_<counter>.png

    # 创建保存路径
    save_vis = os.path.join(save_path, 'zero-shot')

    if not os.path.exists(save_vis):
        os.makedirs(save_vis)

    # 保存可视化图像
    cv2.imwrite(os.path.join(save_vis, filename), vis)

    # 增加计数器
    counter += 1

    return counter  # 返回更新后的计数器


def apply_ad_scoremap(image, scoremap, alpha=0.1):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

