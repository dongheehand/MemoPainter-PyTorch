from torchvision import models
import torch
import torch.nn as nn


class ResNet18(nn.Module):
    def __init__(self, pre_trained = True, require_grad = False):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained = True)       
        
        self.body = [layers for layers in self.model.children()]
        self.body.pop(-1)
        
        self.body = nn.Sequential(*self.body)
        
        if not require_grad:
            for parameter in self.parameters():
                parameter.requires_grad = False
                
    def forward(self, x):
        x = self.body(x)
        x = x.view(-1, 512)
        return x

