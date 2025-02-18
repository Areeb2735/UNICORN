import os
import torch
import torch.nn as nn
from torchvision.models import densenet121
 
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device {device}')
 
model = densenet121(pretrained=True)
model = nn.DataParallel(model)
model.to(device)
input_tensor = torch.randn(128, 3, 224, 224).to(device)
outputs = model(input_tensor)
print(f'Output shape: {outputs.shape}')