import torch
from torch import nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)

    
    
# class DownBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, *args, **kwargs):
#         super().__init__(*args, **kwargs)
        
#         self.layers = nn.Sequential(
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             ConvLayer(in_channels,out_channels),
#         )
        
#     def forward(self,x):
#         x = self.layers(x)
#         return x
    
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ConvLayer(in_channels + out_channels, out_channels,dropout=0.2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = ConvLayer(out_channels * 2, out_channels,dropout=0.2)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
      
    
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64,128,256,512], bilinear=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        for feature in features:
            self.downs.append(ConvLayer(in_channels,feature))
            in_channels = feature
            
        self.bottleneck = ConvLayer(features[-1],features[-1] * 2,dropout=0.4)
        
        for feature in reversed(features):
            self.ups.append(UpBlock(feature * 2,feature,bilinear=bilinear))
            
        self.final_conv = nn.Conv2d(features[0],out_channels,kernel_size=1)
        
    def forward(self,x):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = nn.MaxPool2d(kernel_size=2,stride=2)(x)
            
        x = self.bottleneck(x)
        
        for up, skip in zip(self.ups, reversed(skip_connections)):
            x = up(x, skip)
                
        x = self.final_conv(x)
        
        return x