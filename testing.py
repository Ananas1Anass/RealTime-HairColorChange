# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 11:48:27 2023

@author: User
"""

import torch
import matplotlib.pyplot as plt
from model import UNET
from utils import load_checkpoint
from PIL import Image
from utils import predict_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# parameters
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

model = UNET(in_channels=3, out_channels=1).to(DEVICE)
load_checkpoint(torch.load("my_checkpoint_updated.pth.tar", map_location=torch.device('cpu')), model)
img = Image.open('Anass.jpeg')
predict_image(img, model, device=DEVICE)