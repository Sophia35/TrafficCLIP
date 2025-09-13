# TrafficCLIP 
> [**A DUAL-PATH FRAMEWORK FOR FINE-GRAINED TRAFFIC ACCIDENT DETECTION WITH VISION-LANGUAGE MODELS**]

## abstract 
Existing vision-based traffic accident detection methods typically focus on determining the accident frame or localizing accidents with coarse bounding boxes, limiting detailed pixel-level understanding. To address this, we construct a pixel-level annotated accident dataset and propose TrafficCLIP, a vision-language framework that unifies frame-level accident classification and pixel-level localization. It adopts a dual-path design: one path leverages CLIP’s cross-modal alignment for global accident recognition, while the other employs Local-Aware Attention Mechanism to enhance local spatial localization. The dual-path design separates classification and localization to focus on global and local features respectively, while the shared vision encoder provides unified representations to both paths, enabling task-specific feature extraction and complementary learning. Experiments show that TrafficCLIP achieves a well-balanced trade-off between classification, localization, and real-time performance, while also supporting early anticipation.

## Overview of TrafficCLIP
![overview](https://github.com/Sophia35/TrafficCLIP/blob/main/TrafficCLIP.png)

## How to Run
### Prepare dataset
Download our fine-grained traffic accident dataset from the following Google Drive link and place it under `TrafficCLIP/accident/`:
[Google Drive Link](https://drive.google.com/file/d/1snuQ5fn0FA9rEKhvWrBzKqdnwkxYj_7e/view?usp=sharing)
### Run TrafficCLIP
Quick start (use the pre-trained weights)
1. Download the pre-trained weights train-fine-grained-dataset from [Google Drive Link](https://drive.google.com/file/d/1vZMjZAR9vssg9Ev7e7mQS6BrbVwfwUks/view?usp=sharing)
2. Place the downloaded weights under the directory:`TrafficCLIP/checkpoints/`
3. Run the following command for inference:
```bash
python test.py
```
Run the following command to train TrafficCLIP:
```bash
python train.py
```

## Experiments on CCD Video Dataset
We further validated TrafficCLIP on the CCD video dataset. In this setting, a Transformer with learnable position embeddings was added after the classification path. For evaluation, we report the Average Precision (AP) to measure video-level accident detection performance.
For convenience, we extracted the CLIP features of the CCD dataset and provided them at the following link: [Google Drive link](https://drive.google.com/file/d/1cW6r2ItTpf0pjLc8bpcVvIVlVa-YIL_W/view?usp=sharing)
### Run TrafficCLIP on Video
Quick start (use the pre-trained weights)
1. Download the extracted CCD features and update the corresponding paths in `ccd_feature_train.csv` and `ccd_feature_test.csv`.
2. Download the pre-trained weights train-CCD-video from [Google Drive Link](https://drive.google.com/file/d/1vZMjZAR9vssg9Ev7e7mQS6BrbVwfwUks/view?usp=sharing). Place the downloaded weights under the directory: `TrafficCLIP/checkpoints/`
4. Run inference with:
```bash
python test2_CCD_video_level.py
```
Run the following command to train：
```bash
python train2_CCD_video_level.py
```

