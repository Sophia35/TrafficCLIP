# TrafficCLIP 
> [**A DUAL-PATH FRAMEWORK FOR FINE-GRAINED TRAFFIC ACCIDENT DETECTION WITH VISION-LANGUAGE MODELS**]

## abstract 
Existing vision-based traffic accident detection methods typically focus on determining the accident frame or localizing accidents with coarse bounding boxes, limiting fine-grained understanding.
To address this, we construct a fine-grained accident dataset with accident masks and propose TrafficCLIP, a vision-language framework for accident detection that supports both image-level classification and pixel-level localization. It adopts a dual-path design:  one path leverages CLIP’s cross-modal alignment for accidents recognition, while the other employs Local-Aware Attention Mechanism to enhance localization. The dual-path design separates classification and localization to focus on global and local features respectively, while the shared vision encoder provides unified representations to both paths, enabling task-specific features extraction and complementary learning. Experiments show that TrafficCLIP achieves an optimal balance between classification, localization, and real-time performance, while also supporting early anticipation.

## Overview of TrafficCLIP
![overview](https://github.com/zqhang/AnomalyCLIP/blob/main/assets/overview.png)

## How to Run
### Prepare your dataset
Download our fine-grained traffic accident dataset from the following Google Drive links:
https://drive.google.com/file/d/1snuQ5fn0FA9rEKhvWrBzKqdnwkxYj_7e/view?usp=sharing
### Run TrafficCLIP








* We thank for the code repository: [open_clip](https://github.com/mlfoundations/open_clip), [DualCoOp](https://github.com/sunxm2357/DualCoOp), [CLIP_Surgery](https://github.com/xmed-lab/CLIP_Surgery), and [VAND](https://github.com/ByChelsea/VAND-APRIL-GAN/tree/master).

