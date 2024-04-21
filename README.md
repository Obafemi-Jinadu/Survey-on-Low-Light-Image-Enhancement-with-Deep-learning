# An Impact Study of Deep Learning-based Low-light Image Enhancement in Intelligent Transportation Systems
A repository for the paper 'An Impact Study of Deep Learning-based Low-light Image Enhancement in Intelligent Transportation Systems'

To begin, download the [BDD100K dataset](https://dl.cv.ethz.ch/bdd100k/data/) (Specifically 100k_images_train and bdd100k_det_20_labels_trainval zip files), and run the data parsing code to seeded to reproduce the test images used for this survey.
![image](https://github.com/Obafemi-Jinadu/Survey-on-Low-Light-Image-Enhancement-with-Deep-learning/blob/2620d56bfda48ccbe25c877942c4280b9a62f222/multi%20media%20files/final_img.png)

 | Year | Model | Paper | Learning Strategy| Code|
| ---| --- | --- | --- |--- |
|2021 |SGZ | [Semantic-guided zero-shot learning for low-light image/video enhancement](https://arxiv.org/abs/2110.00970)  | Zero-shot| [code](https://github.com/ShenZheng2000/Semantic-Guided-Low-Light-Image-Enhancement) - PyTorch|
|2021 |Zero-DCE++ | [Learning to Enhance Low-Light Image via Zero-Reference Deep Curve Estimation](https://ieeexplore.ieee.org/document/9369102?denied=) - TPAMI| Zero-shot| [code](https://li-chongyi.github.io/Proj_Zero-DCE++.html) - PyTorch|
|2021 |Zhang et al. | [Learning Temporal Consistency for Low Light Video Enhancement from Single Images](https://ieeexplore.ieee.org/document/9578889) - CVPR| Supervised|[code](https://github.com/zkawfanx/StableLLVE) - PyTorch |
|2021 |EnlightenGAN | [EnlightenGAN: Deep Light Enhancement Without Paired Supervision](https://ieeexplore.ieee.org/document/9334429) -TIP | Unsupervised|[code](https://github.com/VITA-Group/EnlightenGAN) - PyTorch |
|2021 |RetinexDIP | [RetinexDIP: A Unified Deep Framework for Low-Light Image Enhancement](https://ieeexplore.ieee.org/document/9405649) - TCSVT | Zero-shot| [code](https://github.com/zhaozunjin/RetinexDIP) - PyTorch|
|2021 |RUAS | [Retinex-inspired Unrolling with Cooperative Prior Architecture Search for Low-light Image Enhancement](https://www.computer.org/csdl/proceedings-article/cvpr/2021/450900k0556/1yeKx3Rv5ba) - CVPR| Zero-shot|[code](https://github.com/KarelZhang/RUAS) - PyTorch|
|2021 |DRBN | [From Fidelity to Perceptual Quality: A Semi-Supervised Approach for Low-Light Image Enhancement](https://ieeexplore.ieee.org/document/9156559) - CVPR| Semi-supervised|[code](https://github.com/flyywh/CVPR-2020-Semi-Low-Light) - PyTorch |
|2021 |KinD++ | [Beyond Brightening Low-light Images](https://link.springer.com/article/10.1007/s11263-020-01407-x) - IJCV | Supervised|[code](https://github.com/zhangyhuaee/KinD_plus) - TensorFlow |


Zhang et al.,28 KinD++,29 SNR-Aware,21 DPIENet,40 IAT,27 WaveNet,31 URetinex-Net,25 Retinex- Former,41 GlobalDiff,30 PyDiff,42 CDAN,23 PPFormer,43 LYT-Net,22 CIDNet,24 DiffLL.26 Unsupervised learning method: EnlightenGAN,33 SCI,34 UNIE.32 Semi-supervised learning method: DRBN.35 Zero-shot learning method: SGZ,36 RetinexDIP,37 Zero-DCE++38 RUAS39

## 2. Low-Light Images (Real-World)
From left to right, and from top to bottom: Dark, PIE [5], LIME [6], Retinex [1], MBLLEN [7], KinD [2] , Zero-DCE [4], Ours

<p float="left">
<p align="middle">
  <img src="Samples/Dark7.jpg" width="200" />
  <img src="Samples/PIE7.jpg" width="200" /> 
  <img src="Samples/LIME7.jpg" width="200" />
  <img src="Samples/Retinex7.jpg" width="200" />
  <img src="Samples/mbllen7.jpg" width="200" /> 
  <img src="Samples/KinD7.jpg" width="200" />
  <img src="Samples/ZeroDCE7.jpg" width="200" /> 
  <img src="Samples/Ours7.jpg" width="200" />
</p>


| | | | |
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/Obafemi-Jinadu/Survey-on-Low-Light-Image-Enhancement-with-Deep-learning/blob/de2effb13ab47f7ee64cb52602c24861f19106b6/multi%20media%20files/city%20street_clear_a85cad42-337048b5.jpg"> (a) Baseline|  <img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/Obafemi-Jinadu/Survey-on-Low-Light-Image-Enhancement-with-Deep-learning/blob/c84c2af7611627af0bf0b6e90a26e58f6c8e0f75/multi%20media%20files/city%20street_clear_a85cad42-337048b5(2)dpie.png"> (b) DPIENet|<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/Obafemi-Jinadu/Survey-on-Low-Light-Image-Enhancement-with-Deep-learning/blob/bc3e4ccd1b04fff9b8da3780dab6ad1cc0ed6ea6/multi%20media%20files/city%20street_clear_a85cad42-337048b5(2)ppformer.png"> (c) PPFormer |<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/Obafemi-Jinadu/Survey-on-Low-Light-Image-Enhancement-with-Deep-learning/blob/c0909d095bf3e7394e55fe0a73e8167d0e8dbdf7/multi%20media%20files/city%20street_clear_a85cad42-3370sci.png"> (d) SCI 
|<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/Obafemi-Jinadu/Survey-on-Low-Light-Image-Enhancement-with-Deep-learning/blob/10c9af027b645a78f93094b075d019c467838b03/multi%20media%20files/364_normal.png"> (e) GlobalDiff  |  <img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/Obafemi-Jinadu/Survey-on-Low-Light-Image-Enhancement-with-Deep-learning/blob/f4efa7a1fc5f460c9f0a39d3e5665a361bacde2f/multi%20media%20files/city%20street_clear_a85cad42-337048b5(2)retin.png"> (f) RetinexFormer|<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/Obafemi-Jinadu/Survey-on-Low-Light-Image-Enhancement-with-Deep-learning/blob/c0909d095bf3e7394e55fe0a73e8167d0e8dbdf7/multi%20media%20files/city%20street_clear_a85cad42-337048b5(1).jpg"> (g) SGZ |<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/Obafemi-Jinadu/Survey-on-Low-Light-Image-Enhancement-with-Deep-learning/blob/e61f33e78c9ac8f13ce0e0b58f8623f9af93b342/multi%20media%20files/city%20street_clear_a85cad42-337048b5(2)unie.png"> (h) UNIE|
|<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://user-images.githubusercontent.com/297678/29892310-03e92256-8d83-11e7-9b58-986dcb6f702e.png">  |  <img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://user-images.githubusercontent.com/297678/29892310-03e92256-8d83-11e7-9b58-986dcb6f702e.png">|<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://user-images.githubusercontent.com/297678/29892310-03e92256-8d83-11e7-9b58-986dcb6f702e.png">|<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://user-images.githubusercontent.com/297678/29892310-03e92256-8d83-11e7-9b58-986dcb6f702e.png">|



| | | | |
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
|<img alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/Obafemi-Jinadu/Survey-on-Low-Light-Image-Enhancement-with-Deep-learning/blob/de2effb13ab47f7ee64cb52602c24861f19106b6/multi%20media%20files/city%20street_clear_a85cad42-337048b5.jpg"> (a) Baseline|  <img alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/Obafemi-Jinadu/Survey-on-Low-Light-Image-Enhancement-with-Deep-learning/blob/c84c2af7611627af0bf0b6e90a26e58f6c8e0f75/multi%20media%20files/city%20street_clear_a85cad42-337048b5(2)dpie.png"> (b) DPIENet|<img alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/Obafemi-Jinadu/Survey-on-Low-Light-Image-Enhancement-with-Deep-learning/blob/bc3e4ccd1b04fff9b8da3780dab6ad1cc0ed6ea6/multi%20media%20files/city%20street_clear_a85cad42-337048b5(2)ppformer.png"> (c) PPFormer |<img alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/Obafemi-Jinadu/Survey-on-Low-Light-Image-Enhancement-with-Deep-learning/blob/c0909d095bf3e7394e55fe0a73e8167d0e8dbdf7/multi%20media%20files/city%20street_clear_a85cad42-3370sci.png"> (d) SCI 
|<img alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/Obafemi-Jinadu/Survey-on-Low-Light-Image-Enhancement-with-Deep-learning/blob/10c9af027b645a78f93094b075d019c467838b03/multi%20media%20files/364_normal.png"> (e) GlobalDiff  |  <img alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/Obafemi-Jinadu/Survey-on-Low-Light-Image-Enhancement-with-Deep-learning/blob/f4efa7a1fc5f460c9f0a39d3e5665a361bacde2f/multi%20media%20files/city%20street_clear_a85cad42-337048b5(2)retin.png"> (f) RetinexFormer|<img alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/Obafemi-Jinadu/Survey-on-Low-Light-Image-Enhancement-with-Deep-learning/blob/c0909d095bf3e7394e55fe0a73e8167d0e8dbdf7/multi%20media%20files/city%20street_clear_a85cad42-337048b5(1).jpg"> (g) SGZ |<img alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/Obafemi-Jinadu/Survey-on-Low-Light-Image-Enhancement-with-Deep-learning/blob/e61f33e78c9ac8f13ce0e0b58f8623f9af93b342/multi%20media%20files/city%20street_clear_a85cad42-337048b5(2)unie.png"> (h) UNIE|
|<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://user-images.githubusercontent.com/297678/29892310-03e92256-8d83-11e7-9b58-986dcb6f702e.png">  |  <img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://user-images.githubusercontent.com/297678/29892310-03e92256-8d83-11e7-9b58-986dcb6f702e.png">|<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://user-images.githubusercontent.com/297678/29892310-03e92256-8d83-11e7-9b58-986dcb6f702e.png">|<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://user-images.githubusercontent.com/297678/29892310-03e92256-8d83-11e7-9b58-986dcb6f702e.png">|

![image](https://github.com/Obafemi-Jinadu/Survey-on-Low-Light-Image-Enhancement-with-Deep-learning/blob/e27810c05d6cf202708a55925c25408f8e1799ab/multi%20media%20files/image%20a%20new.png)

![image](https://github.com/Obafemi-Jinadu/Survey-on-Low-Light-Image-Enhancement-with-Deep-learning/blob/c79c830c9d87794533eb398d3f034626bf214e7b/multi%20media%20files/image%20b%20new.png)

![image](https://github.com/Obafemi-Jinadu/Survey-on-Low-Light-Image-Enhancement-with-Deep-learning/blob/2fc3c3dad1095193a5255b34d60177a9278847df/multi%20media%20files/image%20c%20newest.png)
 <img src="https://github.com/Obafemi-Jinadu/Survey-on-Low-Light-Image-Enhancement-with-Deep-learning/blob/71cb0ca092a6f4262cf046a3bfaf701cc37e3075/multi%20media%20files/image%20c.png" width="350" height="50">




To generate identical data samples run:

`python 'data parsing/data_parsing_bdd100k.py' --link "" --data "" --dest ""`

| | | |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://user-images.githubusercontent.com/297678/29892310-03e92256-8d83-11e7-9b58-986dcb6f702e.png">  blah |  <img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://user-images.githubusercontent.com/297678/29892310-03e92256-8d83-11e7-9b58-986dcb6f702e.png">|<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://user-images.githubusercontent.com/297678/29892310-03e92256-8d83-11e7-9b58-986dcb6f702e.png">|
|<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://user-images.githubusercontent.com/297678/29892310-03e92256-8d83-11e7-9b58-986dcb6f702e.png">  |  <img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://user-images.githubusercontent.com/297678/29892310-03e92256-8d83-11e7-9b58-986dcb6f702e.png">|<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://user-images.githubusercontent.com/297678/29892310-03e92256-8d83-11e7-9b58-986dcb6f702e.png">|
|<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://user-images.githubusercontent.com/297678/29892310-03e92256-8d83-11e7-9b58-986dcb6f702e.png">  |  <img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://user-images.githubusercontent.com/297678/29892310-03e92256-8d83-11e7-9b58-986dcb6f702e.png">|<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://user-images.githubusercontent.com/297678/29892310-03e92256-8d83-11e7-9b58-986dcb6f702e.png">|


# contact
`obafemi.jinadu@tufts.edu`

# References
[1]


Prof. Panetta's Vision Sensing and Simulations Lab
