import torch.nn as nn
from einops import rearrange

from .utils.convolution import EncoderConvolution, DecoderConvolution
from .vit import ViT


class Encoder(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim):
        super().__init__()
        self.vit_img_dim = img_dim // patch_dim
        
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)    
        )

        self.encoder1 = EncoderConvolution(out_channels, out_channels * 2, stride=2)
        self.encoder2 = EncoderConvolution(out_channels * 2, out_channels * 4, stride=2)
        self.encoder3 = EncoderConvolution(out_channels * 4, out_channels * 8, stride=2)

        
        self.vit = ViT(self.vit_img_dim, out_channels * 8, out_channels * 8,
                       head_num, mlp_dim, block_num, patch_dim=1, classification=False)

        self.post_transformer = nn.Sequential(
            nn.Conv2d(out_channels * 8, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        

    def forward(self, x):
        
        x1 = self.conv_layer(x)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)

        x = self.encoder3(x3)

        x = self.vit(x)
        x = rearrange(x, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim)

        x = self.post_transformer(x)

        return x, x1, x2, x3
    

class Decoder(nn.Module):
    def __init__(self, out_channels, class_num):
        super().__init__()

        self.decoder1 = DecoderConvolution(out_channels * 8, out_channels * 2)
        self.decoder2 = DecoderConvolution(out_channels * 4, out_channels)
        self.decoder3 = DecoderConvolution(out_channels * 2, int(out_channels * 1 / 2))
        self.decoder4 = DecoderConvolution(int(out_channels * 1 / 2), int(out_channels * 1 / 8))

        self.final_conv = nn.Conv2d(int(out_channels * 1 / 8), class_num, kernel_size=1)

    def forward(self, x, x1, x2, x3):
        x = self.decoder1(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder3(x, x1)
        x = self.decoder4(x)
        x = self.final_conv(x)
        return x
    

class TransUNet(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim, class_num):
        super().__init__()
        self.encoder = Encoder(img_dim, in_channels, out_channels,
                               head_num, mlp_dim, block_num, patch_dim)
        self.decoder = Decoder(out_channels, class_num)

    def forward(self, x):
        x, x1, x2, x3 = self.encoder(x)
        x = self.decoder(x, x1, x2, x3)
        return x
