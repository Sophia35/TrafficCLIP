import csv
import torch
import argparse
import torch.nn.functional as F
from matplotlib import pyplot as plt
from mpmath import sigmoid
from torch.utils.tensorboard.summary import video

from prompt_ensemble import TrafficCLIP_PromptLearner
from loss import FocalLoss, BinaryDiceLoss
from utils import normalize
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm
import numpy as np
import os
import random
from utils import get_transform
import TrafficCLIP_lib
torch.cuda.empty_cache()


def get_similarity_map(sm, shape):
    side = int(sm.shape[1] ** 0.5)
    sm = sm.reshape(sm.shape[0], side, side, -1).permute(0, 3, 1, 2)
    sm = torch.nn.functional.interpolate(sm, shape, mode='bilinear')
    sm = sm.permute(0, 2, 3, 1)
    return sm

def compute_similarity(image_features, text_features, t=2):
    prob_1 = image_features[:, :1, :] @ text_features.t()
    b, n_t, n_i, c = image_features.shape[0], text_features.shape[0], image_features.shape[1], image_features.shape[2]
    feats = image_features.reshape(b, n_i, 1, c) * text_features.reshape(1, 1, n_t, c)
    similarity = feats.sum(-1)
    return (similarity/0.07).softmax(-1), prob_1

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_ccd_frame_labels(path_txt):
    video_to_gt = {}
    with open(path_txt, "r") as f:
        for line in f:
            line = line.strip()
            first_comma = line.find(',')
            vid = line[:first_comma]  # 如 "000001"
            # 从第一个逗号后开始找第一个 ']'，表示 frame label 结束
            rest = line[first_comma + 1:]
            end_of_label = rest.find(']') + 1  # 找到 ']' 结束的位置
            label_str = rest[:end_of_label]  # 截取 "[0, 0, ..., 1]"
            frame_labels = eval(label_str)  # 转为列表
            video_to_gt[vid] = frame_labels
    return video_to_gt


def train2(args):
    logger = get_logger(args.save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load CLIP model and prompt learner
    TrafficCLIP_parameters = {"Prompt_length": args.n_ctx}
    model, _ = TrafficCLIP_lib.load("ViT-L/14@336px", device=device, jit=False)
    model = model.float().eval()
    prompt_learner = TrafficCLIP_PromptLearner(model.to("cpu"), TrafficCLIP_parameters)
    checkpoint = torch.load(args.checkpoint_path)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)
    model.to(device)

    # 冻结 ViT 和 PromptLearner 参数
    for param in prompt_learner.parameters():
        param.requires_grad = False

    for param in model.parameters():
        param.requires_grad = False

    frame_position_embeddings = model.frame_position_embeddings
    frame_position_embeddings.weight.requires_grad = True

    # 构建优化器
    optimizer = torch.optim.Adam(
        [frame_position_embeddings.weight],
        lr=args.learning_rate,
        betas=(0.5, 0.999)
    )

    # Prepare dataset list
    with open(args.train_csv_path, "r") as f:
        reader = csv.reader(f)
        video_paths_and_labels = list(reader)

    epoch_losses = []
    wa = 10  # 权重
    bce_loss = torch.nn.BCELoss()
    for epoch in range(args.epoch):
        loss_list = []
        for video_path, label_str in tqdm(video_paths_and_labels):
            label = 1 if label_str == "accident" else 0

            # Load features [100, 1024]
            features = np.load(video_path)
            features = torch.tensor(features, dtype=torch.float32).to(device).unsqueeze(0)
            if features.shape[1] < 50:
                print(f"Skipped (too short): {video_path}")
                continue  # 直接跳过当前视频

            video_features = model.encode_video(features)
            video_features = video_features / video_features.norm(dim=-1, keepdim=True)

            # Encode text prompt
            prompts, tokenized_prompts = prompt_learner(cls_id=None)
            text_features = model.encode_text_learn(prompts, tokenized_prompts).float().to(device)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Compute similarity: [50, 2]
            video_features = video_features.squeeze(0)  # [50, 768]
            logits = video_features @ text_features.T  # [50, 2]
            logits = logits / 0.07

            # Use class 1 similarity score as anomaly score
            pred = logits[:, 1].sigmoid()  # [50]
            a_v = pred.max()  # [scalar]
            label_tensor = torch.tensor(label, dtype=torch.float32, device=device)

            # Compute loss
            if label == 1:
                # CCD数据集中事故固定发生在第40帧（索引39）
                tau = 39
                weights = torch.zeros(50).to(device)
                for t in range(50):
                    delta = max((tau - t) / 10, 0)  #10是帧率
                    weights[t] = torch.exp(torch.tensor(-delta))
                LF = -torch.mean(weights * torch.log(pred + 1e-6))
            else:
                LF = -torch.mean(torch.log(1 - pred + 1e-6))

            # 视频级损失
            LV = bce_loss(a_v, label_tensor)

            # 联合损失
            loss = LF + wa * LV

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        epoch_loss = np.mean(loss_list)
        epoch_losses.append(epoch_loss)

        if (epoch + 1) % args.print_freq == 0:
            logger.info(f"Epoch [{epoch+1}/{args.epoch}] Loss: {np.mean(loss_list):.4f}")

        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, f"epoch_{epoch+1}.pth")
            torch.save({"frame_position_embeddings": model.frame_position_embeddings.state_dict()}, ckp_path)


    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(epoch_losses)+1), epoch_losses, marker='o', label='Train Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_path, "training_loss_curve.png"))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("TrafficCLIP", add_help=True)
    parser.add_argument("--train_data_path", type=str, default="./accident", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./checkpoints/train-CCD-video', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoints/train/epoch_15.pth',
                        help='path to checkpoint')
    parser.add_argument("--train_csv_path", type=str, default='./CCD_feature_train.csv')
    parser.add_argument("--dataset", type=str, default='accident', help="train dataset name")
    parser.add_argument("--n_ctx", type=int, default=12, help="learnable token length")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="mid layer")

    parser.add_argument("--epoch", type=int, default=50, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--image_size", type=int, default=640, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=56, help="random seed")
    args = parser.parse_args()
    setup_seed(args.seed)
    train2(args)
