import torch
import torchvision
from torch2trt import TRTModule
from torch2trt import torch2trt
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import PIL.Image
import numpy as np
import time
import serial

model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2)
model = model.cuda().eval().half()
model.load_state_dict(torch.load('/home/rptech/labcar/best_model_resnet18.pth'))

print("loading resnet 18 model")

# Set device to GPU
device = torch.device('cuda')

from torch2trt import torch2trt

data = torch.zeros((1, 3, 224, 224)).cuda().half()

model_trt = torch2trt(model, [data], fp16_mode=True)


torch.save(model_trt.state_dict(), '/home/rptech/labcar/best_model.pth')

print("saved")

