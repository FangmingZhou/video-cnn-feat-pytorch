import torch

# model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x48d_wsl')

# resnext101_32x48d
# torch.hub.list('facebookresearch/WSL-Images')


# path = '/some/local/path/pytorch/vision'
# model = torch.hub.load(path, 'resnet50', pretrained=True)

model_dir = '/home/zhoufm/VisualSearch/pytorch_models/facebookresearch_WSL-Images'
model_name = 'resnext101_32x48d_wsl'
model = torch.hub.load(model_dir, model_name, source='local')