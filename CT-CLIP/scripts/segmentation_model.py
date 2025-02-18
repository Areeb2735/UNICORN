import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.functional as nnf


class SingleDeconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0)

    def forward(self, x):
        return self.block(x)


class SingleConv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                               padding=((kernel_size - 1) // 2))

    def forward(self, x):
        return self.block(x)


class Conv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv3DBlock(in_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Deconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv3DBlock(in_planes, out_planes),
            SingleConv3DBlock(out_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

class Conv3DBlock_init(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
        
class seg_model(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, embed_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        
        self.decoder0 = \
            nn.Sequential(
                Conv3DBlock_init(input_dim, 32, 3),  
                nn.Upsample(size=(96, 96, 96), mode='trilinear', align_corners=False),
                Conv3DBlock_init(32, 64, 3),
                Conv3DBlock(64, 256, 3)
            )

        self.decoder3 = \
            nn.Sequential(
                Deconv3DBlock(embed_dim, 512),
                Deconv3DBlock(512, 256),
                Deconv3DBlock(256, 128)
            )

        self.decoder9 = \
            Deconv3DBlock(embed_dim, 512)

        self.decoder12_upsampler = \
            SingleDeconv3DBlock(embed_dim, 512)

        self.decoder9_upsampler = \
            nn.Sequential(
                Conv3DBlock(1024, 512),
                Conv3DBlock(512, 512),
                Conv3DBlock(512, 512),
                SingleDeconv3DBlock(512, 256)
            )

        self.decoder0_header = \
            nn.Sequential(
                Conv3DBlock(512, 64),
                # Conv3DBlock(256, 64),
                SingleConv3DBlock(64, output_dim, 1)
            )
    
    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x

    def forward(self, image, hidden_state):

        z0 = image
        z3, z6, z9, z12 = torch.unbind(hidden_state, dim=1)

        z9 = self.proj_feat(z9, 512, (24, 24, 24))
        z12 = self.proj_feat(z12, 512, (24, 24, 24))

        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))
        z0 = self.decoder0(z0)
        output = self.decoder0_header(torch.cat([z0, z9], dim=1))

        return output

class Downsample3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class CLIPSeg3DDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.reduces = nn.ModuleList([nn.Linear(512, 128) for _ in range(4)])
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d_model=128, nhead=4) for _ in range(4)])
        self.trans_conv = nn.ConvTranspose3d(128, 1, kernel_size=3, stride=1, padding=1)
        self.downsample = nn.ModuleList([Downsample3D(in_channels=512, out_channels=512) for _ in range(4)])

        # Example: Multi-stage upsampling block
        self.up1 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv3d(32, 2, kernel_size=1)


    def forward(self, activations):
        
        activations = torch.unbind(activations, dim=1)
        bs = activations[0].shape[0]

        act = None
        for i, (activation, block, reduce, downsample) in enumerate(zip(activations, self.blocks, self.reduces, self.downsample)):
            if act is not None:
                downsampled = downsample(activation)
                bs, channels, D, H, W = downsampled.shape
                tokens = D * H * W
                act = reduce(downsampled.view(bs, tokens, channels)) + act

                # act = reduce(downsample(activation).reshape(bs, 1728, 512)) + act
            else:

                downsampled = downsample(activation)
                bs, channels, D, H, W = downsampled.shape
                tokens = D * H * W
                act = reduce(downsampled.view(bs, tokens, channels))
                # act = reduce(downsample(activation).reshape(bs, 1728, 512))

            act = block(act)

        num_tokens = act.shape[1]
        cube_side = int(round(num_tokens ** (1/3)))

        act = act.view(bs, act.shape[2], cube_side, cube_side, cube_side)

        # act = self.trans_conv(act)

        act = self.up1(act)
        act = self.up2(act)
        act = self.up3(act)
        act = self.final_conv(act)

        # inp_image = torch.randn(bs, 1, 240, 480, 480)
        # act = nnf.interpolate(act, size=inp_image.shape[2:], mode='trilinear', align_corners=True)

        return act


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPSeg3DDecoder()
    model = model.to(device)
    output = model(torch.randn(4, 1, 512, 24, 24, 24).to(device))
    print(output.shape)
