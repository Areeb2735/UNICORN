import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
                nn.Upsample(size=(384, 384, 384), mode='trilinear', align_corners=False),
                Conv3DBlock_init(32, 64, 3)
            )

        self.decoder3 = \
            nn.Sequential(
                Deconv3DBlock(embed_dim, 512),
                Deconv3DBlock(512, 256),
                Deconv3DBlock(256, 128)
            )

        self.decoder6 = \
            nn.Sequential(
                Deconv3DBlock(embed_dim, 512),
                Deconv3DBlock(512, 256),
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

        self.decoder6_upsampler = \
            nn.Sequential(
                Conv3DBlock(512, 256),
                Conv3DBlock(256, 256),
                SingleDeconv3DBlock(256, 128)
            )

        self.decoder3_upsampler = \
            nn.Sequential(
                Conv3DBlock(256, 128),
                Conv3DBlock(128, 128),
                SingleDeconv3DBlock(128, 64)
            )

        self.decoder0_header = \
            nn.Sequential(
                Conv3DBlock(128, 64),
                Conv3DBlock(64, 64),
                SingleConv3DBlock(64, output_dim, 1)
            )

    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x
    
    def forward(self, image, hidden_state):
        z0, z3, z6, z9, z12 = image, *hidden_state

        # z3 = self.proj_feat(z3, 512, (24, 24, 24))
        # z6 = self.proj_feat(z6, 512, (24, 24, 24))
        # z9 = self.proj_feat(z9, 512, (24, 24, 24))
        # z12 = self.proj_feat(z12, 512, (24, 24, 24))
        
        breakpoint()
        z12 = self.decoder12_upsampler(self.proj_feat(z12, 512, (24, 24, 24)))
        z9 = self.decoder9(self.proj_feat(z9, 512, (24, 24, 24)))
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))
        z6 = self.decoder6(self.proj_feat(z6, 512, (24, 24, 24)))
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))
        z3 = self.decoder3(self.proj_feat(z3, 512, (24, 24, 24)))
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))
        z0 = self.decoder0(image)
        output = self.decoder0_header(torch.cat([z0, z3], dim=1))

        # z9 = self.decoder9_upsampler(torch.cat([self.decoder9(self.proj_feat(z9, 512, (24, 24, 24))), self.decoder12_upsampler(self.proj_feat(z12, 512, (24, 24, 24)))], dim=1))
        # z6 = self.decoder6_upsampler(torch.cat([self.decoder6(self.proj_feat(z6, 512, (24, 24, 24))), z9], dim=1))
        # z3 = self.decoder3_upsampler(torch.cat([self.decoder3(self.proj_feat(z3, 512, (24, 24, 24))), z6], dim=1))
        # z0 = self.decoder0(image)
        # output = self.decoder0_header(torch.cat([z0, z3], dim=1))

        return output


if __name__ == "__main__":
    # device = "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = seg_model()
    # model = model.to(device)
    # model(torch.randn(1, 1, 240, 480, 480).to(device), [torch.randn(1, 13824, 512).to(device), torch.randn(1, 13824, 512).to(device), torch.randn(1, 13824, 512).to(device), torch.randn(1, 13824, 512).to(device)])
    model = model.to(device)
    model(torch.randn(1, 1, 240, 480, 480).to(device), [torch.randn(1, 13824, 512).to(device), torch.randn(1, 13824, 512).to(device), torch.randn(1, 13824, 512).to(device), torch.randn(1, 13824, 512).to(device)])

# if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(f'Using device {device}')
    # model = seg_model()
    # model = nn.DataParallel(model)
    # model.to(device)
    #     # Sample input tensors
    # input_image = torch.randn(1, 1, 240, 480, 480).to("cuda")
    # hidden_states = [torch.randn(1, 13824, 512).to("cuda") for _ in range(4)]
    # # model(torch.randn(1, 1, 240, 480, 480).to(device), [torch.randn(1, 13824, 512).to(device), torch.randn(1, 13824, 512).to(device), torch.randn(1, 13824, 512).to(device), torch.randn(1, 13824, 512).to(device)])

    # # Forward pass
    # output = model(input_image, hidden_states)
    # print("Output shape:", output.shape)


    # import os
    # import torch
    # import torch.nn as nn
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(f'Using device {device}')
    
    # model = seg_model()
    # model = nn.DataParallel(model)
    # model.to(device)
    # breakpoint()
    # input_image = torch.randn(1, 1, 240, 480, 480).to(device)
    # hidden_states = [torch.randn(1, 13824, 512).to(device) for _ in range(4)]
    # outputs = model(input_image, hidden_states)
    # print(f'Output shape: {outputs.shape}')
