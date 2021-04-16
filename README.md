# Using Pytorch to Extract 2D CNN Features of Video Frames
This repository is modified from [video-cnn-feature](https://github.com/xuchaoxi/video-cnn-feat).

## Updates
April 16, 2021: unified to RGB mode: convert to RGB mode if the imput image is gray mode.

## Supported Models and Options
### Supported Models:
 - [ResNeXt_WSL](https://github.com/facebookresearch/WSL-Images)

<!-- * …… -->

### Features:
 - oversample: tencrop the input image
 - unified to RGB mode: convert to "RGB" mode if the mode of input image is other modes


### To-do list
 <!-- - universal: add more supported models: -->
 -  [Resnest](https://github.com/zhanghang1989/ResNeSt) (coming soon)


## Environments
* Ubuntu 16.04
* CUDA 10.1
* python 3.8
* torch 1.7.1+cu10
* torchvision 0.8.2+cu101
* Pillow 8.1.2
* numpy 1.20.2

The repository has been tested in the above environment, you don't have to use the same environment, BUT "python=3.6 pytorch >=1.7" is recommended.

This is an example to create a virtual environment using anaconda.
```bash
conda create -n cnn-feat-pytorch python=3.8
conda activate cnn-feat-pytorch
pip install -r requirements 
conda deactivate
```



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
├─VideoData
├─ImageData
└─id.imagepath.txt
```
The `toydata` folder is assumed to be placed at `$HOME/VisualSearch/`. Video files are stored in the `VideoData` folder. Frame files are in the `ImageData`folder. 
+ Video filenames shall end with `.mp4`, `.avi`, `.webm`, or `.gif`.
+ Frame filenames shall end with `.jpg`.

Feature extraction for a given video collection is performed in the following four steps. ***Skip the first step if frames are already there***. 

### Step 1. Extract frames from videos 
Convert the videos (3d) to frames (2d), so that we can employ the 2D CNN models mentioned before.
If you are dealing an image dataset, such as "mscoco", just skip the first step. (Make sure the images are placed in the "ImageData" sub-folder) (Default get one frame every half second)
```
collection=toydata
bash do_extract_frames.sh $collection
```
If you have trouble in extracting frames from *.gif files, use "convert" command in linux as substitute.

### Step 2. Extract frame-level CNN features
Extract the CNN feature of each frame. Results will be placed in the "FeatureData" sub-folder.
```
bash do_wsl-resnext.sh $collection
```

### Step 3. Obtain video-level CNN features (by mean pooling over frames)
```
feature_name=resnext101_32x48d_wsl,avgpool,os
bash do_feature_pooling.sh $collection $feature_name
```

### Step 4. Feature concatenation
If you have more than one features of a collection, this script can combine them into one feature file.
```
featname=$feature_name1+$feature_name2
bash do_concat_features.sh $collection $featname
```

## Acknowledgements
Framework: https://github.com/xuchaoxi/video-cnn-feat

WSL model tutorial: https://pytorch.org/hub/facebookresearch_WSL-Images_resnext

WSL pretrained model: https://github.com/facebookresearch/WSL-Images


