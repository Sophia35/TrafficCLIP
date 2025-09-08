import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble import TrafficCLIP_PromptLearner
from loss import FocalLoss, BinaryDiceLoss
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


def train(args):
    logger = get_logger(args.save_path)
    preprocess, target_transform = get_transform(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    TrafficCLIP_parameters = {"Prompt_length": args.n_ctx}
    model, _ = TrafficCLIP_lib.load("ViT-L/14@336px", device=device, jit=False)
    model = model.float()
    train_data = Dataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    prompt_learner = TrafficCLIP_PromptLearner(model.to("cpu"), TrafficCLIP_parameters)
    prompt_learner.to(device)
    model.to(device)

    optimizer = torch.optim.Adam(list(prompt_learner.parameters()),lr=args.learning_rate, betas=(0.5, 0.999))

    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    model.eval()
    prompt_learner.train()
    for epoch in tqdm(range(args.epoch)):
        model.eval()
        prompt_learner.train()
        loss_list = []
        image_loss_list = []

        for items in tqdm(train_dataloader):
            image = items['img'].to(device)
            label =  items['anomaly']
            gt = items['img_mask'].squeeze().to(device)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            with torch.no_grad():
                image_features, patch_features = model.encode_image(image, args.features_list)
                #print("image feature shape: ",image_features.shape) #[B,768]
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            prompts, tokenized_prompts= prompt_learner(cls_id=None) #[2,77,768]
            text_features = model.encode_text_learn(prompts, tokenized_prompts).float() #[2,768]
            text_features = torch.stack(torch.chunk(text_features, dim = 0, chunks = 2), dim = 1)
            text_features = text_features/text_features.norm(dim=-1, keepdim=True) #[1,2,768]
            text_probs = image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)
            text_probs = text_probs[:, 0, ...]/0.07
            image_loss = F.cross_entropy(text_probs.squeeze(), label.long().cuda())
            image_loss_list.append(image_loss.item())


            similarity_map_list = []
            for idx, patch_feature in enumerate(patch_features):
                patch_feature = patch_feature/ patch_feature.norm(dim = -1, keepdim = True)
                #patch_feature [8,2026,768]
                #text_feaute[0] [2,768]
                similarity, _ = compute_similarity(patch_feature, text_features[0])
                # similarity [8,2026,2]
                similarity_map = get_similarity_map(similarity[:, 1:, :], args.image_size).permute(0, 3, 1, 2)
                # similarity_map [8,2,640,640]
                similarity_map_list.append(similarity_map)

            loss = 0
            for i in range(len(similarity_map_list)):
                loss += 2*loss_focal(similarity_map_list[i], gt)
                loss += loss_dice(similarity_map_list[i][:, 1, :, :], gt)
                loss += loss_dice(similarity_map_list[i][:, 0, :, :], 1-gt)

            optimizer.zero_grad()
            (loss+image_loss).backward()
            optimizer.step()
            loss_list.append(loss.item())
        # logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], loss:{:.4f}, image_loss:{:.4f}'.format(epoch + 1, args.epoch, np.mean(loss_list), np.mean(image_loss_list)))

        # save model
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '.pth')
            torch.save({
                    "prompt_learner": prompt_learner.state_dict(),
            }, ckp_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser("TrafficCLIP", add_help=True)
    parser.add_argument("--train_data_path", type=str, default="./accident", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./checkpoints/train-accidentdataset', help='path to save results')
    parser.add_argument("--dataset", type=str, default='accident', help="train dataset name")
    parser.add_argument("--n_ctx", type=int, default=12, help="learnable token length")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="mid layer")

    parser.add_argument("--epoch", type=int, default=15, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--image_size", type=int, default=640, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=56, help="random seed")
    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)
