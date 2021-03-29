import os
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class ImageDataset(data.Dataset):
    def __init__(self, id_path_file):
        # id_path_file = os.path.join(rootpath, collection, 'id.imagepath.txt')
        data = list(map(str.strip, open(id_path_file).readlines()))
        self.image_ids = [x.split()[0] for x in data]
        self.file_names = [x.split()[1] for x in data]
        self.preprocess = preprocess
        # print(self.image_ids)
        

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        file_name = self.file_names[index]
        image = Image.open(file_name)
        if self.preprocess:
            image = self.preprocess(image)
        # if torch.cuda.is_available():
        #     image = image.to(device)
        return image_id, image

    def __len__(self):
        return len(self.image_ids)



# dataloder = data.DataLoader(dataset, batch_size, shuffle, num_workers)

if __name__ == '__main__':
    rootpath = './VisualSearch'
    collection = 'toydata'
    id_path_file = os.path.join(rootpath, collection, 'id.imagepath.txt')

    dataset = ImageDataset(id_path_file)
    # print(dataset)
    dataloder = data.DataLoader(dataset, batch_size=3, shuffle=False, num_workers=2)
    for i in dataloder:
        # print(*i)
        import pdb; pdb.set_trace(i)
       
        break