import argparse
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
import time


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


def test(args):
    img_size = args.image_size
    features_list = args.features_list
    dataset_dir = args.data_path
    save_path = args.save_path
    dataset_name = args.dataset

    logger = get_logger(args.save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    TrafficCLIP_parameters = {"Prompt_length": args.n_ctx}
    model,_=TrafficCLIP_lib.load("ViT-L/14@336px", device=device, jit=False)
    model = model.float()
    model.eval()

    preprocess, target_transform = get_transform(args)
    test_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list

    results = {}
    metrics = {}
    for obj in obj_list:
        results[obj] = {}
        results[obj]['gt_sp'] = []
        results[obj]['pr_sp'] = []
        results[obj]['imgs_masks'] = []
        results[obj]['anomaly_maps'] = []
        metrics[obj] = {}
        metrics[obj]['pixel-auroc'] = 0
        metrics[obj]['pixel-aupro'] = 0
        metrics[obj]['image-auroc'] = 0
        metrics[obj]['image-ap'] = 0

    prompt_learner = TrafficCLIP_PromptLearner(model.to("cpu"),TrafficCLIP_parameters)
    checkpoint = torch.load(args.checkpoint_path)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])

    prompt_learner.to(device)
    model.to(device)
    prompts, tokenized_prompts= prompt_learner(cls_id=None)
    text_features = model.encode_text_learn(prompts,tokenized_prompts).float()
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    model.to(device)
    #counter=0
    start_time = time.time()
    for idx, items in enumerate(tqdm(test_dataloader)):
        image = items['img'].to(device)
        cls_name = items['cls_name']
        cls_id = items['cls_id']
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name[0]]['imgs_masks'].append(gt_mask)  # px
        results[cls_name[0]]['gt_sp'].extend(items['anomaly'].detach().cpu())

        with torch.no_grad():
            #print("image shape:",image.shape)
            image_features, patch_features = model.encode_image(image, features_list)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            #print("image feature :",image_features.shape)  [1,768]
            #print("text_features:",text_features.shape)    [1,2,768]
            text_probs = image_features @ text_features.permute(0, 2, 1)
            text_probs = (text_probs / 0.07).softmax(-1)
            text_probs = text_probs[:, 0, 1]
            anomaly_map_list = []
            for idx, patch_feature in enumerate(patch_features):
                patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                similarity, _ = compute_similarity(patch_feature, text_features[0])
                similarity_map = get_similarity_map(similarity[:, 1:, :], args.image_size)
                #print(similarity_map.shape)  [1,640,640,2]
                anomaly_map = (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0
                anomaly_map_list.append(anomaly_map)

            anomaly_map = torch.stack(anomaly_map_list)
            anomaly_map = anomaly_map.sum(dim=0)
            results[cls_name[0]]['pr_sp'].extend(text_probs.detach().cpu())
            anomaly_map = torch.stack(
                [torch.from_numpy(gaussian_filter(i, sigma=args.sigma)) for i in anomaly_map.detach().cpu().to(torch.float32)], dim=0)
            results[cls_name[0]]['anomaly_maps'].append(anomaly_map)
            #counter=visualizer(image.detach().cpu().numpy(), anomaly_map.detach().cpu().numpy(), args.image_size, args.save_path,counter,text_probs.detach().cpu(), image_path)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total inference time: {total_time:.4f} seconds")
    print(f"Average inference time per sample: {total_time /len(test_dataloader):.4f} seconds")

    table_ls = []
    image_auroc_list = []
    image_ap_list = []
    pixel_auroc_list = []
    pixel_aupro_list = []
    image_F1_List=[]
    pixel_F1_List=[]
    for obj in obj_list:
        table = []
        table.append(obj)
        results[obj]['imgs_masks'] = torch.cat(results[obj]['imgs_masks'])
        results[obj]['anomaly_maps'] = torch.cat(results[obj]['anomaly_maps']).detach().cpu().numpy()
        if args.metrics == 'image-level':
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            table.append(str(np.round(image_auroc * 100, decimals=1)))
            table.append(str(np.round(image_ap * 100, decimals=1)))
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap)
        elif args.metrics == 'pixel-level':
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            table.append(str(np.round(pixel_aupro * 100, decimals=1)))
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
        elif args.metrics == 'image-pixel-level':
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            image_F1=image_level_metrics(results, obj, "image-F1")
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
            pixel_F1=pixel_level_metrics(results, obj, "pixel-F1")

            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            table.append(str(np.round(pixel_aupro * 100, decimals=1)))
            table.append(str(np.round(image_auroc * 100, decimals=1)))
            table.append(str(np.round(image_ap * 100, decimals=1)))
            table.append(str(np.round(image_F1 * 100, decimals=1)))
            table.append(str(np.round(pixel_F1 * 100, decimals=1)))

            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap)
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
            image_F1_List.append(image_F1)
            pixel_F1_List.append(pixel_F1)
        table_ls.append(table)


    if args.metrics == 'image-level':
        # logger
        table_ls.append(['mean',
                        str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(image_ap_list) * 100, decimals=1))])
        results = tabulate(table_ls, headers=['objects', 'image_auroc', 'image_ap'], tablefmt="pipe")
    elif args.metrics == 'pixel-level':
        # logger
        table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1))
                       ])
        results = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro'], tablefmt="pipe")
    elif args.metrics == 'image-pixel-level':
        # logger
        table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1)),
                        str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(image_ap_list) * 100, decimals=1)),str(np.round(np.mean(image_F1_List) * 100, decimals=1)),str(np.round(np.mean(pixel_F1_List) * 100, decimals=1))])
        results = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro', 'image_auroc', 'image_ap','image_F1',"pixel_F1"], tablefmt="pipe")
    logger.info("\n%s", results)




if __name__ == '__main__':
    parser = argparse.ArgumentParser("TrafficCLIP", add_help=True)
    # paths
    parser.add_argument("--data_path", type=str, default="./accident", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoints/train-fine-grained-dataset/epoch_15.pth',
                        help='path to checkpoint')
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
    test(args)


