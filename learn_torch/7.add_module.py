import torch
import torch.nn as nn
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=True)

print(vgg16)

vgg16.add_module('fc', nn.Linear(1000, 10))

print("=========================================================")
print(vgg16)