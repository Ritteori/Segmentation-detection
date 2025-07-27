import torch
from torch import nn

class ConvLayer(nn.Module):
    def init(self, in_channels, out_channels, kernel_size, stride, padding, n_layers=1, *args, **kwargs):
        super().init(*args, **kwargs)
        
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
    def init(self, in_channels, kernel_size, stride, padding, n_layers, use_residual = True, *args, **kwargs):
        super().init(*args, **kwargs)

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
    def init(self, img_shape:tuple, *args, **kwargs):
        super().init(*args, **kwargs)

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