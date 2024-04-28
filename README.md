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
|2021 |Zero-DCE++ | [Learning to Enhance Low-Light Image via Zero-Reference Deep Curve Estimation](https://ieeexplore.ieee.org/document/9369102?denied=) - `TPAMI 2021`| Zero-shot| [code](https://li-chongyi.github.io/Proj_Zero-DCE++.html) - PyTorch|
|2021 |Zhang et al. | [Learning Temporal Consistency for Low Light Video Enhancement from Single Images](https://ieeexplore.ieee.org/document/9578889) - `CVPR 2021`| Supervised|[code](https://github.com/zkawfanx/StableLLVE) - PyTorch |
|2021 |EnlightenGAN | [EnlightenGAN: Deep Light Enhancement Without Paired Supervision](https://ieeexplore.ieee.org/document/9334429) - `TIP 2021`| Unsupervised|[code](https://github.com/VITA-Group/EnlightenGAN) - PyTorch |
|2021 |RetinexDIP | [RetinexDIP: A Unified Deep Framework for Low-Light Image Enhancement](https://ieeexplore.ieee.org/document/9405649) - `TCSVT 2021` | Zero-shot| [code](https://github.com/zhaozunjin/RetinexDIP) - PyTorch|
|2021 |RUAS | [Retinex-inspired Unrolling with Cooperative Prior Architecture Search for Low-light Image Enhancement](https://www.computer.org/csdl/proceedings-article/cvpr/2021/450900k0556/1yeKx3Rv5ba) - `CVPR 2021`| Zero-shot|[code](https://github.com/KarelZhang/RUAS) - PyTorch|
|2021 |DRBN | [From Fidelity to Perceptual Quality: A Semi-Supervised Approach for Low-Light Image Enhancement](https://ieeexplore.ieee.org/document/9156559) - `TIP 2021`| Semi-supervised|[code](https://github.com/flyywh/CVPR-2020-Semi-Low-Light) - PyTorch |
|2021 |KinD++ | [Beyond Brightening Low-light Images](https://link.springer.com/article/10.1007/s11263-020-01407-x) - `IJCV 2021` | Supervised|[code](https://github.com/zhangyhuaee/KinD_plus) - TensorFlow |
|2021 |SGZ | [Semantic-guided zero-shot learning for low-light image/video enhancement](https://arxiv.org/abs/2110.00970) - `WACV 2022`  | Zero-shot| [code](https://github.com/ShenZheng2000/Semantic-Guided-Low-Light-Image-Enhancement) - PyTorch|
|2022 |DPIENet | [Deep Perceptual Image Enhancement Network for Exposure Restoration](https://ieeexplore.ieee.org/document/9693338) - `TC 2022`  | Supervised| PyTorch|
|2022 |LLFlow | [Low-light image enhancement with normalizing flow](https://wyf0912.github.io/LLFlow/) - AAAI 2022  | Supervised|[code](https://github.com/wyf0912/LLFlow?tab=readme-ov-file) -  PyTorch|
|2022 |SCI | [Toward fast, flexible, and robust low-light image enhancement](https://ieeexplore.ieee.org/document/9879599) - `CVPR 2022`  | Unsupervised|[code](https://github.com/vis-opt-group/SCI) - PyTorch|
|2022 |UNIE | [Unsupervised night image enhancement: When layer decomposition meets light-effects suppression (UNIE)](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136970396.pdf&ved=2ahUKEwic3v_b2tOFAxUxAHkGHa2LDY8QFnoECBUQAQ&usg=AOvVaw1H4dzgTZtPRbo2EDEJpR7c) - `ECCV 2022`  | Unsupervised|[code](https://github.com/jinyeying/night-enhancement) - PyTorch|
|2022 |URetinex-Net | [Uretinex-net: Retinex-based deep unfolding network for low-light image enhancement](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://openaccess.thecvf.com/content/CVPR2022/papers/Wu_URetinex-Net_Retinex-Based_Deep_Unfolding_Network_for_Low-Light_Image_Enhancement_CVPR_2022_paper.pdf&ved=2ahUKEwirhPPI29OFAxXSAHkGHY7BB6UQFnoECAYQAQ&usg=AOvVaw1Ps31kYgQgbCdxemD4ZPQ9) - `CVPR 2022`  | Supervised|[code](https://github.com/AndersonYong/URetinex-Net/tree/main) - PyTorch|
|2022 |SNR-Aware | [SNR-Aware Low-Light Image Enhancement](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_SNR-Aware_Low-Light_Image_Enhancement_CVPR_2022_paper.pdf&ved=2ahUKEwihnp-b3NOFAxXajYkEHTOiCVkQFnoECAYQAQ&usg=AOvVaw2rrjPE2sKpphy2PuQDQjOz) - `CVPR 2022`  | Supervised|[code](https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance) - PyTorch|
|2022 |IAT | [You only need 90k parameters to adapt light: a light weight transformer for image enhancement and exposure correction](https://arxiv.org/abs/2205.14871) - `BMCV 2022`  | Supervised|[code](https://github.com/cuiziteng/Illumination-Adaptive-Transformer?tab=readme-ov-file) - PyTorch|
|2023 |RetinexFormer | [Retinexformer: One-stage retinex-based transformer for low-light image enhancement](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://openaccess.thecvf.com/content/ICCV2023/papers/Cai_Retinexformer_One-stage_Retinex-based_Transformer_for_Low-light_Image_Enhancement_ICCV_2023_paper.pdf&ved=2ahUKEwiEweaN3tOFAxVDlIkEHXmOBQIQFnoECBMQAQ&usg=AOvVaw0LYzHMKZ0cA9DsbRtIVjOx) - `ICCV 2023`  | Supervised|[code](https://github.com/caiyuanhao1998/Retinexformer) - PyTorch|
|2023 |GlobalDiff | [Global structure-aware diffusion process for low-light image enhancement](https://proceedings.neurips.cc/paper_files/paper/2023/hash/fc034d186280f55370b6aca7a3285a65-Abstract-Conference.html) - `NeurIPS 2023`  | Supervised|[code](https://github.com/jinnh/GSAD) - PyTorch|
|2023 |PyDiff | [Pyramid Diffusion Models For Low-light Image Enhancement](https://arxiv.org/pdf/2305.10028.pdf) - `IJCAI 2023`  | Supervised|[code](https://github.com/limuloo/PyDIff) - PyTorch|
|2023 |WaveNet | [WaveNet: Wave-Aware Image Enhancement](https://diglib.eg.org/bitstream/handle/10.2312/pg20231267/021-029.pdf) - PG 2023  | Supervised|[code](https://github.com/DeniJsonC/WaveNet?tab=readme-ov-file) - PyTorch|
|2023 |CDAN | [CDAN: Convolutional dense attention-guided network for low-light image enhancement](https://dl.acm.org/doi/10.1145/3618373) - `ACM TOG 2023`  | Supervised|[code](https://github.com/JianghaiSCU/Diffusion-Low-Light) - PyTorch|
|2024 |PPFormer | [PPformer: Using pixel-wise and patch-wise cross-attention for low-light image enhancement](https://www.sciencedirect.com/science/article/abs/pii/S1077314224000110?via%3Dihub) - CVIU 2024  | Supervised|[code](https://github.com/DeniJsonC/PPformer) - PyTorch|
|2024 |LYT-Net | [LYT-Net: Lightweight YUV Transformer-based Network for Low-Light Image Enhancement](https://arxiv.org/abs/2401.15204) - `2024`  | Supervised|[code](https://github.com/albrateanu/LYT-Net) - PyTorch|
|2024 |CIDNet | [You only need one color space: An efficient network for low-light image enhancement](https://arxiv.org/abs/2402.05809v1) - `2024`  | Supervised|[code](https://github.com/Fediory/HVI-CIDNet?tab=readme-ov-file) - PyTorch|


## Low-Light Image Enhancement Qualitative Outputs Results
From left to right, and from top to bottom: Baseline/Low-light Image [1], DPIENet [2], PPFormer [3],  SCI [4], GlobalDiff [5], RetinexFormer [6], SGZ [7], UNIE [8] sample 1:

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

From left to right, and from top to bottom: Baseline/Low-light Image [1], DPIENet [2], PPFormer [3],  SCI [4], GlobalDiff [5], RetinexFormer [6], SGZ [7], UNIE [8] sample 2:

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




To begin, download the [BDD100K dataset](https://dl.cv.ethz.ch/bdd100k/data/)[1] (Specifically 100k_images_train and bdd100k_det_20_labels_trainval zip files), and run the data parsing code, seeded to reproduce the test images used for this survey, in terminal:
```
python 'data parsing/data_parsing_bdd100k.py' --link "path_to_annotation_bdd100k_file" --data "image_dir" --dest "destination_dir"
```
Alternatively, the sample data used for our experiments can be downloaded [here](https://drive.google.com/drive/folders/1tJkKP505rqO8khdyXHos_lYg4suQpGEN?usp=sharing) rather than running the code above.
# contact
```
obafemi.jinadu@tufts.edu
```

## Non-Reference IQA: BRISQUE & NIQE
```
cd BasicSR
```
- For BRISQUE run the code below:
```
python IQA_metrics.py --niqe_path "./BasicSR/basicsr" --metric BRISQUE --data "path to image folder"
```
- For NIQE run the code below:
```
python IQA_metrics.py --niqe_path "./BasicSR/basicsr" --metric NIQE --data "path to image folder"
```
- For BRISQUE and NIQE run the code below:
```
python IQA_metrics.py --niqe_path "./BasicSR/basicsr" --metric both --data "path to image folder"
```

Note:
* For NIQE, code was adapted from the great work of XPixelGroup's [BasicSR toolbox](https://github.com/XPixelGroup/BasicSR)
* For BRISQUE installation required: ```pip install brisque```
## Citation
```
Available soon
```

# References

[1] Yu, F., Chen, H., Wang, X., Xian, W., Chen, Y., Liu, F., Madhavan, V., and Darrell, T., “Bdd100k: A diverse driving dataset for heterogeneous multitask learning,” in [IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)], (June 2020).

[2] Panetta, K., K. M., S. K., Rao, S. P., and Agaian, S. S., “Deep perceptual image enhancement network for exposure restoration,” IEEE Transactions on Cybernetics 53(7), 4718–4731 (2023).

[3] Dang, J., Zhong, Y., and Qin, X., “PPformer: Using pixel-wise and patch-wise cross-attention for low-light image enhancement,” Computer Vision and Image Understanding 241, 103930 (2024).

[4] Ma, L., Ma, T., Liu, R., Fan, X., and Luo, Z., “Toward fast, flexible, and robust low-light image enhancement,” in [2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)], 5627–5636, IEEE Computer Society, Los Alamitos, CA, USA (jun 2022).

[5] Jinhui Hou, Zhiyu Zhu, J. H. H. L. H. Z. and Yuan, H., “Global structure-aware diffusion process for low-light image enhancement,” Advances in Neural Information Processing Systems (2023).

[6] Cai, Y., Bian, H., Lin, J., Wang, H., Timofte, R., and Zhang, Y., “Retinexformer: One-stage retinex-based transformer for low-light image enhancement,” in [2023 IEEE/CVF International Conference on Computer Vision (ICCV)], 12470–12479, IEEE Computer Society, Los Alamitos, CA, USA (oct 2023).

[7] Zheng, S. and Gupta, G., “Semantic-guided zero-shot learning for low-light image/video enhancement,” in [Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision], 581–590 (2022).

[8] Jin, Y., Yang, W., and Tan, R. T., “Unsupervised night image enhancement: When layer decomposition meets light-effects suppression,” in [European Conference on Computer Vision], 404–421, Springer (2022).

```
Dr. Karen Panetta's Vision and Sensing Lab., Tufts University
```


