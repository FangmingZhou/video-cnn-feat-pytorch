import os,sys
import logging
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from collections import namedtuple

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)

from constant import *

IMG_SIZE = 256
CROP_SIZE = 224
ZERO_IMAGE = np.zeros((IMG_SIZE, IMG_SIZE, 3))
INVALID_ID = 'INVALID'
DEFAULT_OVERSAMPLE = 1
DEVICE_ID = 0

Batch = namedtuple('Batch', ['data'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def img_oversample(raw_img, width=IMG_SIZE, height=IMG_SIZE, crop_dims=CROP_SIZE):
    cropped_image, _ = mx.image.center_crop(raw_img, (crop_dims, crop_dims))
    cropped_image_1 = mx.image.fixed_crop(raw_img, 0, 0, crop_dims, crop_dims)
    cropped_image_2 = mx.image.fixed_crop(raw_img, 0, height-crop_dims, crop_dims, crop_dims)
    cropped_image_3 = mx.image.fixed_crop(raw_img, width-crop_dims, 0, crop_dims, crop_dims)
    cropped_image_4 = mx.image.fixed_crop(raw_img, width-crop_dims, height-crop_dims, crop_dims, crop_dims)
    img_list = [cropped_image.asnumpy(), cropped_image_1.asnumpy(), cropped_image_2.asnumpy(), 
                cropped_image_3.asnumpy(), cropped_image_4.asnumpy()]
    return img_list


def preprocess_images(inputs, width=IMG_SIZE, height=IMG_SIZE, crop_dims=CROP_SIZE,  oversample=True):
    
    

    input_ = [] 
    for ix, input_image in enumerate(inputs):
        raw_img = preprocess(input_image)
        if oversample:
            pass
            # # Generate center, corner, and mirrored crops.
            # input_.extend(img_oversample(raw_img, width, height, crop_dims))
            # input_.extend(img_oversample(mx.nd.flip(raw_img, axis=1), width, height, crop_dims))
        
        input_.append(raw_img)
    

    # return Batch([input_])
    return 


def get_model(model_dir, model_name):
    model = torch.hub.load(model_dir, model_name, source='local')
    print(model)
    return model





def extract_feature(model, layer, batch_size, imset, image_paths, sub_mean=False, oversample=True):
    assert(len(imset)==1)

    image_path = image_paths[0]
    img = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    torch.cuda.empty_cache()

    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)
    if torch.cuda.is_available():
        input_batch = input_batch.to(device)
    
    
    features = [] 
    def hook(self, input, output):
        features.append(output.view(-1).detach().cpu().numpy())

    model.avgpool.register_forward_hook(hook)
    with torch.no_grad():
        model(input_batch)

    # features = model.get_outputs()[0].asnumpy()
    # if oversample:
    #     features = features.reshape((len(features)//10, 10,-1)).mean(1)
    return (imset, features)


if __name__ == '__main__':
    from constant import *
    model_prefix = os.path.join(ROOT_PATH, DEFAULT_MODEL_PREFIX)
    epoch = 0

    sub_mean = False
    oversample = True

    model = get_feat_extractor(model_prefix, gpuid=-1, oversample=oversample)
    imset = str.split('COCO_train2014_000000042196')
    path_imgs = ['%s.jpg'%x for x in imset]
    _, features = extract_feature(model, 1, imset, path_imgs, sub_mean=sub_mean, oversample=oversample)
    print (features.shape)
 
