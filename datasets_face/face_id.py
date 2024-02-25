import os
import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torch.nn.functional as F
import torchvision 
import random
import pickle
import cv2
from models.face_parsing.infer_onehot import *
from pathlib import Path
from transformers import ViTImageProcessor

imagenet_templates_small = [
    'a photo of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'a photo of a clean {}',
    'a photo of a dirty {}',
    'a dark photo of the {}',
    'a photo of my {}',
    'a photo of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'a photo of the {}',
    'a good photo of the {}',
    'a photo of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'a photo of the clean {}',
    'a rendition of a {}',
    'a photo of a nice {}',
    'a good photo of a {}',
    'a photo of the nice {}',
    'a photo of the small {}',
    'a photo of the weird {}',
    'a photo of the large {}',
    'a photo of a cool {}',
    'a photo of a small {}',
    'an illustration of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'an illustration of a clean {}',
    'an illustration of a dirty {}',
    'a dark photo of the {}',
    'an illustration of my {}',
    'an illustration of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'an illustration of the {}',
    'a good photo of the {}',
    'an illustration of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'an illustration of the clean {}',
    'a rendition of a {}',
    'an illustration of a nice {}',
    'a good photo of a {}',
    'an illustration of the nice {}',
    'an illustration of the small {}',
    'an illustration of the weird {}',
    'an illustration of the large {}',
    'an illustration of a cool {}',
    'an illustration of a small {}',
    'a depiction of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'a depiction of a clean {}',
    'a depiction of a dirty {}',
    'a dark photo of the {}',
    'a depiction of my {}',
    'a depiction of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'a depiction of the {}',
    'a good photo of the {}',
    'a depiction of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'a depiction of the clean {}',
    'a rendition of a {}',
    'a depiction of a nice {}',
    'a good photo of a {}',
    'a depiction of the nice {}',
    'a depiction of the small {}',
    'a depiction of the weird {}',
    'a depiction of the large {}',
    'a depiction of a cool {}',
    'a depiction of a small {}',
]


per_img_token_list = ['*']
reg_token_list = ['face']


class FaceIdDataset(Dataset):
    def __init__(self,
                 face_img_path: str = "datasets_face/1.png",
                 image_size: int = 512,
                 vit_path: str = None,
                 flip_p: float = 0.5,
                 face_parsing_model: str = "models/face_parsing/res/cp/79999_iter.pth",
                 **kwargs,
                 ):
        """ """
        super(FaceIdDataset, self).__init__()


        ''' full image list '''

        self.img_list = [face_img_path]

        self.img_list = self.img_list * 1000
        
        self._length = len(self.img_list)

        ''' transform '''
        self.image_size = image_size

        self.trans = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=0.01),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.vit_face_recog_processor = ViTImageProcessor.from_pretrained(vit_path)  # google/vit-base-patch16-224-in21k


        '''get masks'''
        self.face_parsing_model = face_parsing_init(face_parsing_model)
        img = Image.open(self.img_list[0]).convert('RGB')
        self.face_mask, self.hair_mask = face_parsing_evaluate(self.face_parsing_model, img)     


    def __len__(self):
        return self._length

    def __getitem__(self, i):       
        
        example = {}
        ''' image '''
        img_input = Image.open(self.img_list[i]).convert('RGB')
        aligned_img = img_input.crop((64, 70, 440, 446))  # Crop interesting region


        ''' transform '''
        img = self.trans(img_input)
        vit_input = self.vit_face_recog_processor(images=aligned_img, return_tensors="pt")["pixel_values"][0]   
        face_mask_now = self.face_mask.clone().float()                     
        hair_mask_now = self.hair_mask.clone().float()    

            
        if len(face_mask_now.shape) == 2:
            face_mask_now = face_mask_now.unsqueeze(0)
            hair_mask_now = hair_mask_now.unsqueeze(0)

        ''' text '''
        placeholder_string = per_img_token_list[0]
        text = random.choice(imagenet_templates_small).format('%s person' % placeholder_string) 

        if random.random() < 0.5:
            img = torchvision.transforms.functional.hflip(img) 
            vit_input = torchvision.transforms.functional.hflip(vit_input)
            face_mask_now = torchvision.transforms.functional.hflip(face_mask_now)
            hair_mask_now = torchvision.transforms.functional.hflip(hair_mask_now)
            

        crop_size = (img_input.size[0] - img_input.size[0]//4, img_input.size[0] - img_input.size[0]//4)
        original_size = img_input.size

        img, face_mask_now, hair_mask_now = random_crop_and_resize_torch(img, face_mask_now, hair_mask_now, crop_size, original_size)

        img, face_mask_add_bg, hair_mask_add_bg = self._add_bg(img, None, None, face_mask_now = face_mask_now, hair_mask_now = hair_mask_now)    

        face_mask_add_bg64 = F.interpolate(face_mask_add_bg.unsqueeze(0), (64, 64), mode='bicubic', align_corners=True)[0]
        face_mask_add_bg64[face_mask_add_bg64 >= 0.5] = 1
        face_mask_add_bg64[face_mask_add_bg64 < 0.5] = 0
        
        hair_mask_add_bg64 = F.interpolate(hair_mask_add_bg.unsqueeze(0), (64, 64), mode='bicubic', align_corners=True)[0]
        hair_mask_add_bg64[hair_mask_add_bg64 >= 0.5] = 1
        hair_mask_add_bg64[hair_mask_add_bg64 < 0.5] = 0        
        
        example["pixel_values"] = img
        example["vit_input"] = vit_input    
        example["caption"] = text
        example["face_mask"] = face_mask_add_bg64
        example["hair_mask"] = hair_mask_add_bg64       

        return example

    @staticmethod
    def _add_bg(tensor_img: torch.Tensor, tensor_bg: torch.Tensor = None, scale: list = None, 
                face_mask_now: torch.Tensor = None, hair_mask_now: torch.Tensor = None, 
        ):
        c, h, w = tensor_img.shape
        ret = torch.ones_like(tensor_img, device=tensor_img.device) * -1.  # (C,H,W)
        ret = tensor_bg if tensor_bg is not None else ret
        ret_mask = torch.zeros_like(face_mask_now, device=face_mask_now.device)
        ret_mask_hair = torch.zeros_like(hair_mask_now, device=hair_mask_now.device)

        if scale is None:
            scale = [0.5, 1.0]
        rh = min(int(h * np.random.uniform(scale[0], scale[1])), h)
        rw = min(int(rh * np.random.uniform(0.9, 1.1)), w)
        
        tensor_img = F.interpolate(tensor_img.unsqueeze(0), (rh, rw), mode='bicubic', align_corners=True)
        face_mask_now_resize = F.interpolate(face_mask_now.unsqueeze(0), (rh, rw), mode='bicubic', align_corners=True) 
        hair_mask_now_resize = F.interpolate(hair_mask_now.unsqueeze(0), (rh, rw), mode='bicubic', align_corners=True) 

        pos_h, pos_w = 0, 0
        if h > rh:
            pos_h = np.random.randint(h - rh)
        if w > rw:
            pos_w = np.random.randint(w - rw)

        ret[:, pos_h: pos_h + rh, pos_w: pos_w + rw] = tensor_img[0]
        ret_mask[:, pos_h: pos_h + rh, pos_w: pos_w + rw] = face_mask_now_resize[0]
        ret_mask_hair[:, pos_h: pos_h + rh, pos_w: pos_w + rw] = hair_mask_now_resize[0]
        
        return ret, ret_mask, ret_mask_hair


def tensor_to_arr(tensor):
    return ((tensor + 1.) * 127.5).cpu().numpy().astype(np.uint8)



def random_crop_and_resize_torch(image, face_mask_now, hair_mask_now, crop_size, original_size):

    _, img_height, img_width = image.shape

    left = torch.randint(0, img_width - crop_size[1] + 1, (1,))
    top = torch.randint(0, img_height - crop_size[0] + 1, (1,))

    cropped_image = image[:, top.item():top.item() + crop_size[0], left.item():left.item() + crop_size[1]]
    cropped_face_mask = face_mask_now[:, top.item():top.item() + crop_size[0], left.item():left.item() + crop_size[1]]
    cropped_hair_mask = hair_mask_now[:, top.item():top.item() + crop_size[0], left.item():left.item() + crop_size[1]]
    

    resized_image = F.interpolate(cropped_image.unsqueeze(0), size=original_size, mode='bicubic', align_corners=True)[0]
    
    resized_cropped_face_mask = F.interpolate(cropped_face_mask.unsqueeze(0), size=original_size, mode='bicubic', align_corners=True)[0]
    resized_cropped_hair_mask = F.interpolate(cropped_hair_mask.unsqueeze(0), size=original_size, mode='bicubic', align_corners=True)[0]


    return resized_image, resized_cropped_face_mask, resized_cropped_hair_mask