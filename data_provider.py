import os
import torch
from torch.utils import data
from PIL import Image
import numpy as np
import torchvision
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, TenCrop, Lambda, ToTensor, Normalize
from utils.generic_utils import Progbar
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

oversample_preprocess = Compose([
    Resize(256),
    TenCrop(224),# this is a list of PIL Images
    Lambda(lambda crops: torch.stack([Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(ToTensor()(crop)) for crop in crops])) # returns a 4D tensor
])


preprocess = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Using the mean and std of the ImageNet dataset.
])


class ImageDataset(data.Dataset):
    
    def __init__(self, id_path_file, oversample=False):
        # id_path_file = os.path.join(rootpath, collection, 'id.imagepath.txt')
        data = list(map(str.strip, open(id_path_file).readlines()))
        self.image_ids = [x.split()[0] for x in data]
        self.file_names = [x.split()[1] for x in data]
        if oversample:
            self.preprocess = oversample_preprocess
        else:
            self.preprocess = preprocess
        # print(self.image_ids)
        

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        file_name = self.file_names[index]
        image = Image.open(file_name)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = self.preprocess(image)
        return image_id, image


    def __len__(self):
        return len(self.image_ids)



# dataloder = data.DataLoader(dataset, batch_size, shuffle, num_workers)

if __name__ == '__main__':
    rootpath = './VisualSearch'
    collection = 'toydata'
    id_path_file = os.path.join(rootpath, collection, 'id.imagepath.txt')

    # dataset = ImageDataset(id_path_file)
    # # print(dataset)
    # dataloder = data.DataLoader(dataset, batch_size=3, shuffle=False, num_workers=2)
    # for i in dataloder:
    #     # print(*i)
    #     import pdb; pdb.set_trace(i)
       
    #     break
    
    data = list(map(str.strip, open(id_path_file).readlines()))
    image_ids = [x.split()[0] for x in data]
    file_names = [x.split()[1] for x in data]
    # preprocess = oversample_preprocess

    image_id = image_ids[0]
    file_name = file_names[0]
    image = Image.open(file_name)
    import pdb;pdb.set_trace()

    # image.save('image_example/%s.jpg'%image_id)
    # process = Compose([
    #     Resize(256),
    #     TenCrop(224)# this is a list of PIL Images
    #     # Lambda(lambda crops: torch.stack([Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])()(ToTensor()(crop)) for crop in crops])) # returns a 4D tensor
    # ])
    resized_image = Resize((256,256))(image)
    import pdb; pdb.set_trace()
    resized_image.save('image_example/%s_resized.jpg'%(image_id))

    image_crops = TenCrop(224)(resized_image)
    # import pdb; pdb.set_trace()

    for i, image in enumerate(image_crops):
        image.save('image_example/%s_%d.jpg'%(image_id, i))

    

# # from torchvision import transforms
# unloader = transforms.ToPILImage()
# image = original_tensor.cpu().clone()  # clone the tensor
# image = image.squeeze(0)  # remove the fake batch dimension
# image = unloader(image)
# image.save('example.jpg')
