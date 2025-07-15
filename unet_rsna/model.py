import torch
from torch import nn

class ConvLayer(nn.Module):
    def __init__(self, in_features, out_channels,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.layers = nn.Sequential(
            
            nn.Conv2d(in_features,out_channels,3,1,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels,out_channels,3,1,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
        )
        
    def forward(self,x):
        return self.layers(x)
    
# class UNet(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
        
#     def forward(self,x):
        