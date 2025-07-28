import torch
from torch import nn

class ConvLayer(nn.Module):    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, n_layers=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.layers = nn.ModuleList()

        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2)))
            in_channels = out_channels

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self,x):

        for module in self.layers:
            
            x = module(x)

        return x
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding, n_layers, use_residual = True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_residual = use_residual
        self.layers = nn.ModuleList()

        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 2,kernel_size, stride, padding),
                nn.LeakyReLU(0.2),
                nn.Conv2d(in_channels // 2, in_channels ,kernel_size, stride, padding),
                nn.BatchNorm2d(in_channels)
            ))
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self,x):

        for layer  in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)

        return x
    

class DarkNet53(nn.Module):
    def __init__(self, img_shape:tuple, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.large = nn.Sequential(

            ConvLayer(img_shape[0],32,kernel_size=3,stride=1,padding=1), # 416 x 416 x 32
            ConvLayer(32,64,kernel_size=3,stride=2,padding=1), # 208 x 208 x 64

            ResidualBlock(64,kernel_size=3,stride=1,padding=1,n_layers=1,use_residual=True), # 208 x 208 x 64

            ConvLayer(64,128,kernel_size=3,stride=2,padding=1), # 104 x 104 x 128
            ResidualBlock(128,kernel_size=3,stride=1,padding=1,n_layers=2,use_residual=True), # 104 x 104 x 128

            ConvLayer(128,256,kernel_size=3,stride=2,padding=1), # 52 x 52 x 256
            ResidualBlock(256,kernel_size=3,stride=1,padding=1,n_layers=8,use_residual=True), # 52 x 52 x 256

        )

        self.medium = nn.Sequential(

            ConvLayer(256,512,kernel_size=3,stride=2,padding=1), # 26 x 26 x 512
            ResidualBlock(512,kernel_size=3,stride=1,padding=1,n_layers=8,use_residual=True), # 26 x 26 x 512

        )

        self.small = nn.Sequential(

            ConvLayer(512,1024,kernel_size=3,stride=2,padding=1), # 13 x 13 x 1024
            ResidualBlock(1024,kernel_size=3,stride=1,padding=1,n_layers=4,use_residual=True), # 13 x 13 x 1024

            ConvLayer(1024,1024,kernel_size=1,stride=1,padding=0)

        )

    def forward(self,x):
        large = self.large(x)
        medium = self.medium(large)
        small = self.small(medium)
        
        return small, medium, large
    
class YOLOv3(nn.Module):
    def __init__(self, img_shape:tuple, num_anchors=3, num_classes=20, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone = DarkNet53(img_shape)
        
        self.small_conv = nn.Sequential(
            ConvLayer(1024, 512, kernel_size=1, stride=1, padding=0),
            ConvLayer(512, 1024, kernel_size=3, stride=1, padding=1),
            ConvLayer(1024, 512, kernel_size=1, stride=1, padding=0),
            ConvLayer(512, 1024, kernel_size=3, stride=1, padding=1),
            ConvLayer(1024, 512, kernel_size=1, stride=1, padding=0)
        )
        self.pred_small = nn.Conv2d(512, num_anchors*(5 + num_classes), kernel_size=1)
        
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        
        self.medium_conv = nn.Sequential(
            ConvLayer(768, 256, kernel_size=1, stride=1, padding=0),
            ConvLayer(256, 512, kernel_size=3, stride=1, padding=1),
            ConvLayer(512, 256, kernel_size=1, stride=1, padding=0),
            ConvLayer(256, 512, kernel_size=3, stride=1, padding=1),
            ConvLayer(512, 256, kernel_size=1, stride=1, padding=0)
        )
        self.pred_medium = nn.Conv2d(256, num_anchors*(5 + num_classes), kernel_size=1) # tx, ty, tw, th, obj_score + probs
        
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        
        self.large_conv = nn.Sequential(
            ConvLayer(384, 128, kernel_size=1, stride=1, padding=0),
            ConvLayer(128, 256, kernel_size=3, stride=1, padding=1),
            ConvLayer(256, 128, kernel_size=1, stride=1, padding=0),
            ConvLayer(128, 256, kernel_size=3, stride=1, padding=1),
            ConvLayer(256, 128, kernel_size=1, stride=1, padding=0)
        )
        self.pred_large = nn.Conv2d(128, num_anchors*(5 + num_classes), kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight,nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        small, medium, large = self.backbone(x)
        
        x_small = self.small_conv(small)
        out_small = self.pred_small(x_small)

        x_up = self.upsample1(x_small)  # 13x13x512 -> 26x26x256
        x_medium = torch.cat([x_up, medium], dim=1) # 26x26x768
        x_medium = self.medium_conv(x_medium) # 26x26x256
        out_medium = self.pred_medium(x_medium) 
        
        x_up = self.upsample2(x_medium)  # 26x26x256 -> 52x52x128
        x_large = torch.cat([x_up, large], dim=1) # 52x52x384
        x_large = self.large_conv(x_large) # 52x52x128
        out_large = self.pred_large(x_large)

        return out_small, out_medium, out_large
    