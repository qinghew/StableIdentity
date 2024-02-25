#!/usr/bin/python
# -*- encoding: utf-8 -*-

from models.face_parsing.model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
from PIL import Image
import random

def face_parsing_init(model_addr):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.load_state_dict(torch.load(model_addr))
    net.eval()
    return net

def zero_padding(img, size0, pad1, pad2):
    zero_padding = np.zeros((img.shape[0], size0, size0), dtype=np.float32)
    pad1 = pad1 / 2
    pad2 = pad2 / 2
    zero_padding[:, int(pad1):int(size0 - pad1), int(pad2):int(size0 - pad2)] = img[:,:,:]

    return zero_padding



def parsing_2_masks(parsing_anno):

    get_face = parsing_anno.copy().astype(np.uint8) 
    get_components = parsing_anno.copy().astype(np.uint8) 
    
    get_hair = parsing_anno.copy().astype(np.uint8) 
    get_hair[get_hair == 17] = 255
    get_hair[get_hair < 128] = 0
    hair_mask = torch.from_numpy(get_hair.astype(np.uint8))       
    
    get_face[get_face == 9] = 0
    get_face[get_face == 14] = 0
    get_face[get_face == 15] = 0
    get_face[get_face == 16] = 0
    get_face[get_face == 17] = 0
    get_face[get_face == 18] = 0
    
    get_face[get_face>0] = 1

    face_mask = torch.from_numpy(get_face.astype(np.uint8))       

    return face_mask, hair_mask


def face_parsing_evaluate(net, img):

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():
        
        img = to_tensor(img)
        img = torch.unsqueeze(img, 0)
        img = img.cuda() 

        out = net(img)[0]      
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        face_mask, hair_mask = parsing_2_masks(parsing)
        
    return face_mask, hair_mask




