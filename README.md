<h1 align="center">An Impact Study of Deep Learning-based Low-light Image Enhancement in Intelligent Transportation Systems</h1>


The repository for our paper **An Impact Study of Deep Learning-based Low-light Image Enhancement in Intelligent Transportation Systems**: [Obafemi Jinadu](https://femi-jinadu.github.io/), [Srijith Rajeev](https://scholar.google.com/citations?user=9vac4DkAAAAJ&hl=en), [Karen A. Panetta](https://scholar.google.com/citations?user=nsOodtAAAAAJ&hl=en), [Sos Agaian](https://scholar.google.com/citations?user=FazfMZMAAAAJ&hl=en)

## Hightlights
-  Comprehensive and Updated Review of deep learning-based LLIE techniques. Including transformer-based and diffusion-based models.
 -   The performance of deep learning-based LLIE methods are evaluated specifically within the context of Intelligent Transportation Systems (ITS). 
  - Impact on Downstream Computer Vision Tasks such as Object Detection.
    
![image](https://github.com/Obafemi-Jinadu/Survey-on-Low-Light-Image-Enhancement-with-Deep-learning/blob/2620d56bfda48ccbe25c877942c4280b9a62f222/multi%20media%20files/final_img.png)



## Abstract
Images and videos captured in poor illumination conditions are degraded by low brightness, reduced contrast, color distortion, and noise, rendering them barely discernable for human perception and ultimately negatively impacting computer vision system performance. These challenges are exasperated when processing video surveillance camera footage, using this unprocessed video data as-is for real-time computer vision tasks across varying environmental conditions within Intelligent Transportation Systems (ITS), such as vehicle detection, tracking, and timely incident detection. 
The inadequate performance of these algorithms in real-world deployments incurs significant operational costs. Low-light image enhancement (LLIE) aims to improve the quality of images captured in these unideal conditions. Groundbreaking advancements in LLIE have been recorded employing deep-learning techniques to address these challenges, however, the plethora of models and approaches is varied and disparate. This paper presents an exhaustive survey to explore a methodical taxonomy of state-of-the-art deep learning-based LLIE algorithms and their impact when used in tandem with other computer vision algorithms, particularly detection algorithms. To thoroughly evaluate these LLIE models, a subset of the BDD100K dataset, a diverse real-world driving dataset is used for suitable image quality assessment and evaluation metrics. This study aims to provide a detailed understanding of the dynamics between low-light image enhancement and ITS performance, offering insights into both the technological advancements in LLIE and their practical implications in real-world conditions.

 | Year | Model | Paper - Publication| Learning Strategy| Code|
| ---| --- | --- | --- |--- |
|2021 |SGZ | [Semantic-guided zero-shot learning for low-light image/video enhancement](https://arxiv.org/abs/2110.00970) - WACV 2022  | Zero-shot| [code](https://github.com/ShenZheng2000/Semantic-Guided-Low-Light-Image-Enhancement) - PyTorch|
|2021 |Zero-DCE++ | [Learning to Enhance Low-Light Image via Zero-Reference Deep Curve Estimation](https://ieeexplore.ieee.org/document/9369102?denied=) - TPAMI| Zero-shot| [code](https://li-chongyi.github.io/Proj_Zero-DCE++.html) - PyTorch|
|2021 |Zhang et al. | [Learning Temporal Consistency for Low Light Video Enhancement from Single Images](https://ieeexplore.ieee.org/document/9578889) - CVPR| Supervised|[code](https://github.com/zkawfanx/StableLLVE) - PyTorch |
|2021 |EnlightenGAN | [EnlightenGAN: Deep Light Enhancement Without Paired Supervision](https://ieeexplore.ieee.org/document/9334429) -TIP | Unsupervised|[code](https://github.com/VITA-Group/EnlightenGAN) - PyTorch |
|2021 |RetinexDIP | [RetinexDIP: A Unified Deep Framework for Low-Light Image Enhancement](https://ieeexplore.ieee.org/document/9405649) - TCSVT | Zero-shot| [code](https://github.com/zhaozunjin/RetinexDIP) - PyTorch|
|2021 |RUAS | [Retinex-inspired Unrolling with Cooperative Prior Architecture Search for Low-light Image Enhancement](https://www.computer.org/csdl/proceedings-article/cvpr/2021/450900k0556/1yeKx3Rv5ba) - CVPR| Zero-shot|[code](https://github.com/KarelZhang/RUAS) - PyTorch|
|2021 |DRBN | [From Fidelity to Perceptual Quality: A Semi-Supervised Approach for Low-Light Image Enhancement](https://ieeexplore.ieee.org/document/9156559) - CVPR| Semi-supervised|[code](https://github.com/flyywh/CVPR-2020-Semi-Low-Light) - PyTorch |
|2021 |KinD++ | [Beyond Brightening Low-light Images](https://link.springer.com/article/10.1007/s11263-020-01407-x) - IJCV | Supervised|[code](https://github.com/zhangyhuaee/KinD_plus) - TensorFlow |


Zhang et al.,28 KinD++,29 SNR-Aware,21 DPIENet,40 IAT,27 WaveNet,31 URetinex-Net,25 Retinex- Former,41 GlobalDiff,30 PyDiff,42 CDAN,23 PPFormer,43 LYT-Net,22 CIDNet,24 DiffLL.26 Unsupervised learning method: EnlightenGAN,33 SCI,34 UNIE.32 Semi-supervised learning method: DRBN.35 Zero-shot learning method: SGZ,36 RetinexDIP,37 Zero-DCE++38 RUAS39

## Low-Light Image Enhancement Qualitative Outputs Results
From left to right, and from top to bottom: Baseline/Low-light Image, DPIENet [], PPFormer [],  SCI [], GlobalDiff [], RetinexFormer [], SGZ [], UNIE [] sample 1:

<p float="left">
<p align="middle">
  <img src="https://github.com/Obafemi-Jinadu/Survey-on-Low-Light-Image-Enhancement-with-Deep-learning/blob/de2effb13ab47f7ee64cb52602c24861f19106b6/multi%20media%20files/city%20street_clear_a85cad42-337048b5.jpg" width="200"/>
  <img src="https://github.com/Obafemi-Jinadu/Survey-on-Low-Light-Image-Enhancement-with-Deep-learning/blob/c84c2af7611627af0bf0b6e90a26e58f6c8e0f75/multi%20media%20files/city%20street_clear_a85cad42-337048b5(2)dpie.png" width="200"/> 
  <img src="https://github.com/Obafemi-Jinadu/Survey-on-Low-Light-Image-Enhancement-with-Deep-learning/blob/bc3e4ccd1b04fff9b8da3780dab6ad1cc0ed6ea6/multi%20media%20files/city%20street_clear_a85cad42-337048b5(2)ppformer.png" width="200"/>
  <img src="https://github.com/Obafemi-Jinadu/Survey-on-Low-Light-Image-Enhancement-with-Deep-learning/blob/c0909d095bf3e7394e55fe0a73e8167d0e8dbdf7/multi%20media%20files/city%20street_clear_a85cad42-3370sci.png" width="200"/>
  <img src="https://github.com/Obafemi-Jinadu/Survey-on-Low-Light-Image-Enhancement-with-Deep-learning/blob/10c9af027b645a78f93094b075d019c467838b03/multi%20media%20files/364_normal.png" width="200"/> 
  <img src="https://github.com/Obafemi-Jinadu/Survey-on-Low-Light-Image-Enhancement-with-Deep-learning/blob/f4efa7a1fc5f460c9f0a39d3e5665a361bacde2f/multi%20media%20files/city%20street_clear_a85cad42-337048b5(2)retin.png" width="200"/>
  <img src="https://github.com/Obafemi-Jinadu/Survey-on-Low-Light-Image-Enhancement-with-Deep-learning/blob/2384f4ecd86a694c60a4a1d9585223dc083c43bc/multi%20media%20files/city%20street_clear_a85cad42-337048b5(1).jpg" width="200"/> 
  <img src="https://github.com/Obafemi-Jinadu/Survey-on-Low-Light-Image-Enhancement-with-Deep-learning/blob/e61f33e78c9ac8f13ce0e0b58f8623f9af93b342/multi%20media%20files/city%20street_clear_a85cad42-337048b5(2)unie.png" width="200"/>
</p>

From left to right, and from top to bottom: Baseline/Low-light Image, DPIENet [], PPFormer [],  SCI [], GlobalDiff [], RetinexFormer [], SGZ [], UNIE [] sample 1:

<p float="left">
<p align="middle">
  <img src="multi media files/city street_clear_9c70dfb5-10bbd85c(1).jpg" width="200"/>
  <img src="multi media files/city street_clear_9c70dfb5-10bbd85c(2).png" width="200"/> 
  <img src="multi media files/city street_clear_9c70dfb5-10bbd85c(6)_ppformer.png" width="200"/>
  <img src="multi media files/city street_clear_9c70dfb5-10bb_sci.png" width="200"/>
  <img src="multi media files/116_normal_gsad.png" width="200"/> 
  <img src="multi media files/city street_clear_9c70dfb5-10bbd85c(6)_retinexformer.png" width="200"/>
  <img src="multi media files/city street_clear_9c70dfb5-10bbd85c(2)sgz.jpg" width="200"/> 
  <img src="multi media files/city street_clear_9c70dfb5-10bbd85c(6)unie.png" width="200"/>
</p>



## Impact of LLIE on Object Detection
![image](https://github.com/Obafemi-Jinadu/Survey-on-Low-Light-Image-Enhancement-with-Deep-learning/blob/e27810c05d6cf202708a55925c25408f8e1799ab/multi%20media%20files/image%20a%20new.png)

![image](https://github.com/Obafemi-Jinadu/Survey-on-Low-Light-Image-Enhancement-with-Deep-learning/blob/c79c830c9d87794533eb398d3f034626bf214e7b/multi%20media%20files/image%20b%20new.png)

![image](https://github.com/Obafemi-Jinadu/Survey-on-Low-Light-Image-Enhancement-with-Deep-learning/blob/2fc3c3dad1095193a5255b34d60177a9278847df/multi%20media%20files/image%20c%20newest.png)
 <img src="https://github.com/Obafemi-Jinadu/Survey-on-Low-Light-Image-Enhancement-with-Deep-learning/blob/71cb0ca092a6f4262cf046a3bfaf701cc37e3075/multi%20media%20files/image%20c.png" width="350" height="50">




To begin, download the [BDD100K dataset](https://dl.cv.ethz.ch/bdd100k/data/) (Specifically 100k_images_train and bdd100k_det_20_labels_trainval zip files), and run the data parsing code, seeded to reproduce the test images used for this survey, in terminal:
```
python 'data parsing/data_parsing_bdd100k.py' --link "path_to_annotation_bdd100k_file" --data "image_dir" --dest "destination_dir"
```

# contact
```
obafemi.jinadu@tufts.edu
```

# References
[1]


Prof. Panetta's Vision Sensing and Simulations Lab
