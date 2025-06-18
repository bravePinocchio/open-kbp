""" Neural net architectures - Fixed Recursion Version """
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    """Standalone generator class to avoid recursion"""
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        
        # Access DataShapes attributes properly
        # Data dimensions: NDHWC
        ct_channels = parent.data_shapes.ct[-1]
        roi_channels = parent.data_shapes.structure_masks[-1]
        in_channels = ct_channels + roi_channels
        
        # Downsampling path
        self.conv1 = parent.make_convolution_block(in_channels, parent.initial_number_of_filters)
        self.conv2 = parent.make_convolution_block(parent.initial_number_of_filters, 2 * parent.initial_number_of_filters)
        self.conv3 = parent.make_convolution_block(2 * parent.initial_number_of_filters, 4 * parent.initial_number_of_filters)
        self.conv4 = parent.make_convolution_block(4 * parent.initial_number_of_filters, 8 * parent.initial_number_of_filters)
        self.conv5 = parent.make_convolution_block(8 * parent.initial_number_of_filters, 8 * parent.initial_number_of_filters)
        self.conv6 = parent.make_convolution_block(8 * parent.initial_number_of_filters, 8 * parent.initial_number_of_filters, use_batch_norm=False)
        
        # Upsampling path
        self.upconv5 = parent.make_convolution_transpose_block(8 * parent.initial_number_of_filters, 8 * parent.initial_number_of_filters, use_dropout=False)
        self.upconv4 = parent.make_convolution_transpose_block(16 * parent.initial_number_of_filters, 8 * parent.initial_number_of_filters)
        self.upconv3 = parent.make_convolution_transpose_block(16 * parent.initial_number_of_filters, 4 * parent.initial_number_of_filters, use_dropout=False)
        self.upconv2 = parent.make_convolution_transpose_block(8 * parent.initial_number_of_filters, 2 * parent.initial_number_of_filters)
        self.upconv1 = parent.make_convolution_transpose_block(4 * parent.initial_number_of_filters, parent.initial_number_of_filters, use_dropout=False)
        
        # Final layers
        self.final_conv = nn.ConvTranspose3d(
            2*parent.initial_number_of_filters,
            1,
            kernel_size=parent.filter_size,
            stride=parent.stride_size,
            padding=tuple((k-1)//2 for k in parent.filter_size),
            output_padding=tuple(s-2 for s in parent.stride_size)
        )
        self.avg_pool = nn.AvgPool3d(kernel_size=3, stride=1, padding=1)

    def forward(self, ct: torch.Tensor, roi_masks: torch.Tensor) -> torch.Tensor:
        # Ensure proper channel dimensions
        # NDHWC -> NCDHW
        ct = ct.permute(0, 4, 1, 2, 3)
        roi_masks = roi_masks.permute(0, 4, 1, 2, 3)
        
        x = torch.cat([ct, roi_masks], dim=1)
        
        
        # Downsampling path
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        
        
        # Upsampling path
        x5b = self.upconv5(x6)
        x4b = self.upconv4(torch.cat([x5b, x5], dim=1))
        x3b = self.upconv3(torch.cat([x4b, x4], dim=1))
        x2b = self.upconv2(torch.cat([x3b, x3], dim=1))
        x1b = self.upconv1(torch.cat([x2b, x2], dim=1))
        
        # Final processing
        x0b = torch.cat([x1b, x1], dim=1)
        x0b = self.final_conv(x0b)
        x_final = self.avg_pool(x0b)
        
        return F.relu(x_final)

class DefineDoseFromCT:
    """Main class without recursive definition"""
    def __init__(
        self,
        data_shapes: 'DataShapes',
        initial_number_of_filters: int,
        filter_size: Tuple[int, int, int],
        stride_size: Tuple[int, int, int],
        gen_optimizer: torch.optim.Optimizer,
    ):
        self.data_shapes = data_shapes
        self.initial_number_of_filters = initial_number_of_filters
        self.filter_size = filter_size
        self.stride_size = stride_size
        self.gen_optimizer = gen_optimizer
        self.padding = tuple((k-1)//2 for k in self.filter_size)

    def make_convolution_block(self, in_channels: int, out_channels: int, use_batch_norm: bool = True) -> nn.Sequential:
        padding = self.padding
        layers = [
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=self.filter_size,
                stride=self.stride_size,
                padding=padding,
                bias=not use_batch_norm
            )
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm3d(out_channels, momentum=0.99, eps=1e-3))
        layers.append(nn.LeakyReLU(negative_slope=0.2))
        return nn.Sequential(*layers)

    def make_convolution_transpose_block(
        self, in_channels: int, out_channels: int, use_dropout: bool = True
    ) -> nn.Sequential:
        layers = [
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=self.filter_size,
                stride=self.stride_size,
                padding=self.padding,
                output_padding=tuple(s-2 for s in self.stride_size),
                bias=False
            ),
            nn.BatchNorm3d(out_channels, momentum=0.99, eps=1e-3)
        ]
        if use_dropout:
            layers.append(nn.Dropout3d(0.2))
        layers.append(nn.LeakyReLU(negative_slope=0))
        return nn.Sequential(*layers)

    def define_generator(self) -> nn.Module:
        """Returns the generator model"""
        return Generator(self)