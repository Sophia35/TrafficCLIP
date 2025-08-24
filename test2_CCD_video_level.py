import argparse
import csv
import os

import torch
import numpy as np
import random
from logger import get_logger
import TrafficCLIP_lib
from utils import get_transform
from dataset import Dataset
from tqdm import tqdm
from tabulate import tabulate
from scipy.ndimage import gaussian_filter
from metrics import image_level_metrics, pixel_level_metrics
from prompt_ensemble import TrafficCLIP_PromptLearner
from visualization import visualizer
import time
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def evaluation(all_pred, all_labels, time_of_accidents, fps=20.0):
    """
    :param: all_pred (N x T), where N is number of videos, T is the number of frames for each video
    :param: all_labels (N,)
    :param: time_of_accidents (N,) int element
    :output: AP (average precision, AUC), mTTA (mean Time-to-Accident), TTA@R80 (TTA at Recall=80%)
    """

    preds_eval = []
    min_pred = np.inf
    n_frames = 0
    for idx, toa in enumerate(time_of_accidents):
        if all_labels[idx] > 0:
            pred = all_pred[idx, :int(toa)]  # positive video
        else:
            pred = all_pred[idx, :]  # negative video
        # find the minimum prediction
        min_pred = np.min(pred) if min_pred > np.min(pred) else min_pred
        preds_eval.append(pred)
        n_frames += len(pred)
    total_seconds = all_pred.shape[1] / fps

    # iterate a set of thresholds from the minimum predictions
    # temp_shape = int((1.0 - max(min_pred, 0)) / 0.001 + 0.5)
    Precision = np.zeros((n_frames))
    Recall = np.zeros((n_frames))
    Time = np.zeros((n_frames))
    cnt = 0
    for Th in np.arange(max(min_pred, 0), 1.0, 0.001):
        Tp = 0.0
        Tp_Fp = 0.0
        Tp_Tn = 0.0
        time = 0.0
        counter = 0.0  # number of TP videos
        # iterate each video sample
        for i in range(len(preds_eval)):
            # true positive frames: (pred->1) * (gt->1)
            tp =  np.where(preds_eval[i]*all_labels[i]>=Th)
            Tp += float(len(tp[0])>0)
            if float(len(tp[0])>0) > 0:
                # if at least one TP, compute the relative (1 - rTTA)
                time += tp[0][0] / float(time_of_accidents[i])
                counter = counter+1
            # all positive frames
            Tp_Fp += float(len(np.where(preds_eval[i]>=Th)[0])>0)
        if Tp_Fp == 0:  # predictions of all videos are negative
            continue
        else:
            Precision[cnt] = Tp/Tp_Fp
        if np.sum(all_labels) ==0: # gt of all videos are negative
            continue
        else:
            Recall[cnt] = Tp/np.sum(all_labels)
        if counter == 0:
            continue
        else:
            Time[cnt] = (1-time/counter)
        cnt += 1
    # sort the metrics with recall (ascending)
    new_index = np.argsort(Recall)
    Precision = Precision[new_index]
    Recall = Recall[new_index]
    Time = Time[new_index]
    # unique the recall, and fetch corresponding precisions and TTAs
    _,rep_index = np.unique(Recall,return_index=1)
    rep_index = rep_index[1:]
    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    for i in range(len(rep_index)-1):
         new_Time[i] = np.max(Time[rep_index[i]:rep_index[i+1]])
         new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i+1]])
    # sort by descending order
    new_Time[-1] = Time[rep_index[-1]]
    new_Precision[-1] = Precision[rep_index[-1]]
    new_Recall = Recall[rep_index]
    # compute AP (area under P-R curve)
    AP = 0.0
    if new_Recall[0] != 0:
        AP += new_Precision[0]*(new_Recall[0]-0)
    for i in range(1,len(new_Precision)):
        AP += (new_Precision[i-1]+new_Precision[i])*(new_Recall[i]-new_Recall[i-1])/2

    # transform the relative mTTA to seconds
    mTTA = np.mean(new_Time) * total_seconds
    print("Average Precision= %.4f, mean Time to accident= %.4f"%(AP, mTTA))
    sort_time = new_Time[np.argsort(new_Recall)]
    sort_recall = np.sort(new_Recall)
    TTA_R80 = sort_time[np.argmin(np.abs(sort_recall-0.8))] * total_seconds
    print("Recall@80%, Time to accident= " +"{:.4}".format(TTA_R80))

    return AP, mTTA, TTA_R80


def test2_CCD(args):
    logger = get_logger(args.save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and prompt learner
    TrafficCLIP_parameters = {"Prompt_length": args.n_ctx}
    model, _ = TrafficCLIP_lib.load("ViT-L/14@336px", device=device, jit=False)
    model = model.float().eval()
    prompt_learner = TrafficCLIP_PromptLearner(model.to("cpu"), TrafficCLIP_parameters)

    # 加载 checkpoint
    checkpoint1 = torch.load(args.checkpoint_path1, map_location=device)
    checkpoint2 = torch.load(args.checkpoint_path2, map_location=device)
    prompt_learner.load_state_dict(checkpoint1["prompt_learner"])
    model.frame_position_embeddings.load_state_dict(checkpoint2["frame_position_embeddings"])
    prompt_learner.to(device)
    model.to(device)

    # 加载测试集路径
    with open(args.test_csv_path, "r") as f:
        reader = csv.reader(f)
        video_paths_and_labels = list(reader)

    all_preds = []  # shape: (N, T)
    all_gts = []    # shape: (N,)
    toa_list = []   # 每个正样本的事故帧（tau）

    with torch.no_grad():
        for video_path, label_str in tqdm(video_paths_and_labels):
            label = 1 if label_str == "accident" else 0

            # 加载视频特征
            features = np.load(video_path)
            video_features = torch.tensor(features, dtype=torch.float32).to(device).unsqueeze(0)  # [1, T, 1024]
            if features.shape[1] < 50:
                print(f"Skipped (too short): {video_path}")
                continue  # 直接跳过当前视频
            video_features = model.encode_video(video_features)
            video_features = video_features / video_features.norm(dim=-1, keepdim=True)

            # 文本提示编码
            prompts, tokenized_prompts = prompt_learner(cls_id=None)
            text_features = model.encode_text_learn(prompts, tokenized_prompts).float().to(device)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # 相似度匹配
            video_features = video_features.squeeze(0)  # [T, dim]
            logits = video_features @ text_features.T  # [T, 2]
            logits = logits / 0.07
            pred = logits[:, 1].sigmoid().cpu().numpy()  # [T]

            all_preds.append(pred)
            all_gts.append(label)
            toa_list.append(40 if label == 1 else 0)  # CCD 数据集中，40帧表示事故发生帧

    # 转为 numpy 数组用于评估
    all_preds_np = np.stack([np.pad(p, (0, 50 - len(p))) if len(p) < 50 else p[:50] for p in all_preds])  # shape: [N, 50]
    all_gts_np = np.array(all_gts)
    toa_np = np.array(toa_list)
    # 评估 AP, mTTA, TTA@80%
    AP, mTTA, TTA_R80 = evaluation(all_preds_np, all_gts_np, toa_np, fps=10.0)

    logger.info(f"Average Precision (AP): {AP:.4f}")
    logger.info(f"Mean Time to Accident (mTTA): {mTTA:.4f} seconds")
    logger.info(f"Time to Accident at Recall 80% (TTA@R80): {TTA_R80:.4f} seconds")



if __name__ == '__main__':
    parser = argparse.ArgumentParser("TrafficCLIP", add_help=True)
    # paths
    parser.add_argument("--data_path", type=str, default="./accident", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/', help='path to save results')
    parser.add_argument("--checkpoint_path1", type=str, default='./checkpoints/train/epoch_15.pth',
                        help='path to checkpoint')
    parser.add_argument("--checkpoint_path2", type=str, default='./checkpoints/train-CCD-video/epoch_1.pth',
                        help='path to checkpoint')
    parser.add_argument("--test_csv_path", type=str, default='./CCD_feature_test.csv')
    # model
    parser.add_argument("--dataset", type=str, default='accident')
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="mid layer")
    parser.add_argument("--image_size", type=int, default=640, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="learnable token length")
    parser.add_argument("--metrics", type=str, default='image-pixel-level')
    parser.add_argument("--seed", type=int, default=666, help="random seed")
    parser.add_argument("--sigma", type=int, default=4, help="")

    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    test2_CCD(args)