import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
# from image import *
import torchvision.transforms.functional as F
import cv2


class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4):
        if train:
            root = root *4
        random.shuffle(root)
        
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        
    def __len__(self):
        return self.nSamples

    def read_img(self, imgPath):
        labelPath = imgPath.replace('images', 'densitys').replace('png','npy')
        img = cv2.imread(imgPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640,640))
        target = np.load(labelPath)
        # target = cv2.resize(target,(target.shape[1]/8,target.shape[0]/8),interpolation = cv2.INTER_CUBIC)*64
        target = cv2.resize(target,(80,80))*64
        target = np.expand_dims(target, 0)
        return img,target

    def __getitem__(self, index):
        assert index <= len(self), 'index range error' 
        img_path = self.lines[index]
        # img,target = load_data(img_path,self.train)
        img,target = self.read_img(img_path)
        #img = 255.0 * F.to_tensor(img)
        #img[0,:,:]=img[0,:,:]-92.8207477031
        #img[1,:,:]=img[1,:,:]-95.2757037428
        #img[2,:,:]=img[2,:,:]-104.877445883
        if self.transform is not None:
            img = self.transform(img)
        return img,target