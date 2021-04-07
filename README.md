# Using Pytorch to Extract 2D CNN Features of Video Frame


# Supported Models and Options
## Models:
* [ResNeXt_WSL](https://github.com/facebookresearch/WSL-Images)

* ……

## Options:
<!-- * oversample -->

# to do list
 - oversample
 - universal
 - [Resnest](https://github.com/zhanghang1989/ResNeSt)


# Environments
* Ubuntu 16.04
* CUDA 10.1
* python 3.8
* torch 1.7.1+cu10
* torchvision 0.8.2+cu101
* Pillow 8.1.2
* numpy 1.20.2




<!-- # Extracting CNN features from video frames by MXNet

The `video-cnn-feat` toolbox provides python code and scripts for extracting CNN features from video frames by pre-trained [MXNet](http://mxnet.incubator.apache.org/) models. We have used this toolbox for our [winning solution](https://www-nlpir.nist.gov/projects/tvpubs/tv18.papers/rucmm.pdf) at TRECVID 2018 ad-hoc video search (AVS) task and in our [W2VV++](https://dl.acm.org/citation.cfm?doid=3343031.3350906) paper.

## Requirements

### Environments

* Ubuntu 16.04
* CUDA 9.0
* python 2.7
* opencv-python
* mxnet-cu90 
* numpy

We used virtualenv to setup a deep learning workspace that supports MXNet. Run the following script to install the required packages.
```
virtualenv --system-site-packages ~/cnn_feat
source ~/cnn_feat/bin/activate
pip install -r requirements.txt
deactivate
``` -->

<!-- ### MXNet models

#### 1. ResNet-152 from the MXNet model zoo

```
# Download resnet-152 model pre-trained on imagenet-11k
./do_download_resnet152_11k.sh

# Download resnet-152 model pre-trained on imagenet-1k
./do_download_resnet152_1k.sh
```

#### 2. ResNeXt-101 from MediaMill, University of Amsterdam

Send a request to `xirong ATrucDOTeduDOTcn` for the model link. Please read the [ImageNet Shuffle](https://dl.acm.org/citation.cfm?id=2912036) paper for technical details. -->

## Get started

Our code assumes the following data organization. We provide the `toydata` folder as an example.
```
collection_name
+ VideoData
+ ImageData
+ id.imagepath.txt
```
The `toydata` folder is assumed to be placed at `$HOME/VisualSearch/`. Video files are stored in the `VideoData` folder. Frame files are in the `ImageData`folder. 
+ Video filenames shall end with `.mp4`, `.avi`, `.webm`, or `.gif`.
+ Frame filenames shall end with `.jpg`.

Feature extraction for a given video collection is performed in the following four steps. ***Skip the first step if frames are already there***. 

### Step 1. Extract frames from videos 


```
collection=toydata
bash do_extract_frames.sh $collection
```

### Step 2. Extract frame-level CNN features

```
bash do_wsl-resnext.sh $collection
```

### Step 3. Obtain video-level CNN features (by mean pooling over frames)
```
feature_name=resnext101_32x48d_wsl,avgpool,os

bash do_feature_pooling.sh $collection $feature_name
```

### Step 4. Feature concatenation
```
featname=$feature_name1+$feature_name2
bash do_concat_features.sh $collection $featname
```




<!-- # Acknowledgements

This project was supported by the National Natural Science Foundation of China (No. 61672523).

## References
If you find the package useful, please consider citing our MM'19 paper:
```
@inproceedings{li2019w2vv++,
title={W2VV++: Fully Deep Learning for Ad-hoc Video Search},
author={Li, Xirong and Xu, Chaoxi and Yang, Gang and Chen, Zhineng and Dong, Jianfeng},
booktitle={Proceedings of the 27th ACM International Conference on Multimedia},
pages={1786--1794},
year={2019}
}
```
# video-cnn-feat-pytorch -->

# Acknowledgements
https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/

https://github.com/facebookresearch/WSL-Images

https://github.com/xuchaoxi/video-cnn-feat
