import cv2
import os


def video_to_frames(video_path, output_folder):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # 获取视频的帧率（FPS）
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Frames per second: {fps}")

    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 逐帧读取视频并保存为图像
    frame_count = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break  # 如果没有读取到帧，说明视频结束

        # 保存当前帧为图片
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

        # 显示进度
        print(f"Processing frame {frame_count}")
        frame_count += 1

    # 释放视频对象
    cap.release()
    print(f"Finished processing {frame_count} frames.")


# 调用函数
video_path = ""  # 替换为视频文件的路径
output_folder = ""  # 替换为你想保存帧图像的文件夹路径
video_to_frames(video_path, output_folder)