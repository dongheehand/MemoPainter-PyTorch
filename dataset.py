import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from util import NNEncode, encode_313bin
from colorthief import ColorThief


class mydata(Dataset):
    def __init__(self, img_path, img_size, km_file_path, color_info, transform = None, NN = 10.0, sigma = 5.0):
        
        self.img_path = img_path
        self.img_size = img_size
        self.img = sorted(os.listdir(self.img_path))
        
        self.res_normalize_mean = [0.485, 0.456, 0.406]
        self.res_normalize_std = [0.229, 0.224, 0.225]
        self.transform = transform
        self.color_info = color_info
        if self.color_info == 'dist':
            self.nnenc = NNEncode(NN, sigma, km_filepath = km_file_path)
        
    def __len__(self):
        
        return len(self.img)
        
    def __getitem__(self, i):
        
        img_item = {}
        rgb_image = Image.open(os.path.join(self.img_path, self.img[i])).convert('RGB')
        w, h = rgb_image.size
        if w != h:
            min_val = min(w, h)
            rgb_image = rgb_image.crop((w // 2 - min_val // 2, h // 2 - min_val // 2, w // 2 + min_val // 2, h // 2 + min_val // 2))
        
        rgb_image = np.array(rgb_image.resize((self.img_size, self.img_size), Image.LANCZOS))
        
        lab_image = rgb2lab(rgb_image)
        l_image = lab_image[:,:,:1]
        ab_image = lab_image[:,:,1:]
        
        if self.color_info == 'dist':
            color_feat = encode_313bin(np.expand_dims(ab_image, axis = 0), self.nnenc)[0]
            color_feat = np.mean(color_feat, axis = (0, 1))
        
        elif self.color_info == 'RGB':
            color_thief = ColorThief(os.path.join(self.img_path, self.img[i]))
            
            ## color_feat shape : (10, 3)
            color_feat = np.array(color_thief.get_palette(11))
            
            ## color_feat shape : (3, 10)
            color_feat = np.transpose(color_feat)
            
            ## color_feat shape : (30)
            color_feat = np.reshape(color_feat, (30)) / 255.0
            del color_thief
        
        gray_image = [lab_image[:,:,:1]]
        h, w, c = lab_image.shape
        gray_image.append(np.zeros(shape = (h, w, 2)))
        gray_image = np.concatenate(gray_image, axis = 2)
        
        res_input = lab2rgb(gray_image)
        res_input = (res_input - self.res_normalize_mean) / self.res_normalize_std
        
        index = i + 0.0

        img_item['l_channel'] = np.transpose(l_image, (2, 0, 1)).astype(np.float32)
        img_item['ab_channel'] = np.transpose(ab_image, (2, 0, 1)).astype(np.float32)
        img_item['color_feat'] = color_feat.astype(np.float32)
        img_item['res_input'] = np.transpose(res_input, (2, 0, 1)).astype(np.float32)
        img_item['index'] = np.array(([index])).astype(np.float32)[0]
        
            
        return img_item

