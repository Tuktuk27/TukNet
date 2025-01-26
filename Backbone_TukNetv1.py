import torch
import torch.nn as nn
import torch.nn.functional
import torchvision.models

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv


class BACKBONE_Shuffle_relu(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.backbone2 = nn.Sequential(*list(torchvision.models.shufflenet_v2_x1_0(pretrained=True).children())[:-4])
        for p in self.backbone2.parameters():
            p.requires_grad = True
        self.backbone3 = nn.Sequential(*list(torchvision.models.shufflenet_v2_x1_0(pretrained=True).children())[-4:-3])
        for p in self.backbone3.parameters():
            p.requires_grad = True
        self.backbone4 = nn.Sequential(*list(torchvision.models.shufflenet_v2_x1_0(pretrained=True).children())[-3:-2]) 
        for p in self.backbone4.parameters():
            p.requires_grad = True


        self.neck_3_1 = ConvDPUnit(464, 232, True)
        self.neck_3_2 = ConvDPUnit(232, 64, True)
        
        self.neck_2_1 = ConvDPUnit(232, 116, True)
        self.neck_2_2 = ConvDPUnit(116, 64, True)

        self.neck_1_1 = ConvDPUnit(116, 64, True)
        self.neck_1_2 = ConvDPUnit(64, 64, True)

    def forward(self, x):
        # get backbone features
        trained_backbone2 = self.backbone2(x)
        trained_backbone3 = self.backbone3(trained_backbone2)
        trained_backbone4 = self.backbone4(trained_backbone3)
        
        neck_3_1 = self.neck_3_1(trained_backbone4)
        neck_3_2 = self.neck_3_2(neck_3_1)

        
        #upsampled_neck_3_1 = F.interpolate(neck_3_1, size=trained_backbone3.shape[2:], mode='nearest')
        upsampled_neck_3_1 = torch.nn.functional.interpolate(neck_3_1, scale_factor=2., mode='nearest')
        add_4_3_output = upsampled_neck_3_1 + trained_backbone3
        neck_2_1 = self.neck_2_1(add_4_3_output)
        neck_2_2 = self.neck_2_2(neck_2_1)

        #upsampled_neck_2_1 = F.interpolate(neck_2_1, size=backbone_2_2.shape[2:], mode='nearest')
        upsampled_neck_2_1 = torch.nn.functional.interpolate(neck_2_1, scale_factor=2., mode='nearest')
        add_3_2_output = trained_backbone2 + upsampled_neck_2_1 
        neck_1_1 = self.neck_1_1(add_3_2_output)
        neck_1_2 = self.neck_1_2(neck_1_1)

        outputs = (neck_1_2, neck_2_2, neck_3_2)
        return outputs


class ConvDPUnit(nn.Module):
    def __init__(self, in_channels, out_channels, withBNRelu=True):
        super(ConvDPUnit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True, groups=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, groups=out_channels)
        self.withBNRelu = withBNRelu
        if withBNRelu:
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if hasattr(self, 'bn'):
            x = self.bn(x)
            x = self.relu(x)
        return x

