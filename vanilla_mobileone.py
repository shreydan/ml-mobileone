"""
simple mobileone architecture without any reparameterization tricks.
"""
from typing import Optional, List, Tuple

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze and Excite module.

    Pytorch implementation of `Squeeze-and-Excitation Networks` -
    https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self, in_channels: int, rd_ratio: float = 0.0625) -> None:
        """Construct a Squeeze and Excite Module.

        :param in_channels: Number of input channels.
        :param rd_ratio: Input channel reduction ratio.
        """
        super(SEBlock, self).__init__()
        self.reduce = nn.Conv2d(
            in_channels=in_channels,
            out_channels=int(in_channels * rd_ratio),
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.expand = nn.Conv2d(
            in_channels=int(in_channels * rd_ratio),
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""
        b, c, h, w = inputs.size()
        x = F.avg_pool2d(inputs, kernel_size=[h, w])
        x = self.reduce(x)
        x = F.relu(x)
        x = self.expand(x)
        x = torch.sigmoid(x)
        x = x.view(-1, c, 1, 1)
        return inputs * x


class MobileOneBlock(nn.Module):
    """MobileOne building block.

    This block has a multi-branched architecture at train-time
    and plain-CNN style architecture at inference time
    For more details, please refer to our paper:
    `An Improved One millisecond Mobile Backbone` -
    https://arxiv.org/pdf/2206.04040.pdf
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        inference_mode: bool = False,
        use_se: bool = False,
        num_conv_branches: int = 1,
    ) -> None:
        """Construct a MobileOneBlock module.

        :param in_channels: Number of channels in the input.
        :param out_channels: Number of channels produced by the block.
        :param kernel_size: Size of the convolution kernel.
        :param stride: Stride size.
        :param padding: Zero-padding size.
        :param dilation: Kernel dilation factor.
        :param groups: Group number.
        :param inference_mode: If True, instantiates model in inference mode.
        :param use_se: Whether to use SE-ReLU activations.
        :param num_conv_branches: Number of linear conv branches.
        """
        super(MobileOneBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches

        # Check if SE-ReLU is requested
        if use_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = nn.Identity()
        self.activation = nn.ReLU()

        if inference_mode:
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
            )
        else:

            self.skip = (
                nn.BatchNorm2d(num_features=in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )

            conv = list()
            for _ in range(self.num_conv_branches):
                conv.append(self._conv_bn(kernel_size=kernel_size, padding=padding))
            self.conv = nn.ModuleList(conv)

            self.scale = None
            if kernel_size > 1:
                self.scale = self._conv_bn(kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""
        # Inference mode forward pass.
        if self.inference_mode:
            return self.activation(self.se(self.conv(x)))

        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.skip is not None:
            identity_out = self.skip(x)

        # Scale branch output
        scale_out = 0
        if self.scale is not None:
            scale_out = self.scale(x)

        # Other branches
        out = scale_out + identity_out
        for ix in range(self.num_conv_branches):
            out += self.conv[ix](x)

        return self.activation(self.se(out))

    def _conv_bn(self, kernel_size: int, padding: int) -> nn.Sequential:
        """Helper method to construct conv-batchnorm layers.

        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module(
            "conv",
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                padding=padding,
                groups=self.groups,
                bias=False,
            ),
        )
        mod_list.add_module("bn", nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list


class MobileOne(nn.Module):
    """MobileOne Model

    Pytorch implementation of `An Improved One millisecond Mobile Backbone` -
    https://arxiv.org/pdf/2206.04040.pdf
    """

    def __init__(
        self,
        num_blocks_per_stage: List[int] = [2, 8, 10, 1],
        num_classes: int = 1000,
        width_multipliers: Optional[List[float]] = None,
        inference_mode: bool = False,
        use_se: bool = False,
        num_conv_branches: int = 1,
    ) -> None:
        """Construct MobileOne model.

        :param num_blocks_per_stage: List of number of blocks per stage.
        :param num_classes: Number of classes in the dataset.
        :param width_multipliers: List of width multiplier for blocks in a stage.
        :param inference_mode: If True, instantiates model in inference mode.
        :param use_se: Whether to use SE-ReLU activations.
        :param num_conv_branches: Number of linear conv branches.
        """
        super().__init__()

        assert len(width_multipliers) == 4
        self.inference_mode = inference_mode
        self.in_planes = min(64, int(64 * width_multipliers[0]))
        self.use_se = use_se
        self.num_conv_branches = num_conv_branches

        # Build stages
        self.stage0 = MobileOneBlock(
            in_channels=3,
            out_channels=self.in_planes,
            kernel_size=3,
            stride=2,
            padding=1,
            inference_mode=self.inference_mode,
        )
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(
            int(64 * width_multipliers[0]), num_blocks_per_stage[0], num_se_blocks=0
        )
        self.stage2 = self._make_stage(
            int(128 * width_multipliers[1]), num_blocks_per_stage[1], num_se_blocks=0
        )
        self.stage3 = self._make_stage(
            int(256 * width_multipliers[2]),
            num_blocks_per_stage[2],
            num_se_blocks=int(num_blocks_per_stage[2] // 2) if use_se else 0,
        )
        self.stage4 = self._make_stage(
            int(512 * width_multipliers[3]),
            num_blocks_per_stage[3],
            num_se_blocks=num_blocks_per_stage[3] if use_se else 0,
        )
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width_multipliers[3]), num_classes)

    def _make_stage(
        self, planes: int, num_blocks: int, num_se_blocks: int
    ) -> nn.Sequential:
        """Build a stage of MobileOne model.

        :param planes: Number of output channels.
        :param num_blocks: Number of blocks in this stage.
        :param num_se_blocks: Number of SE blocks in this stage.
        :return: A stage of MobileOne model.
        """
        # Get strides for all layers
        strides = [2] + [1] * (num_blocks - 1)
        blocks = []
        for ix, stride in enumerate(strides):
            use_se = False
            if num_se_blocks > num_blocks:
                raise ValueError(
                    "Number of SE blocks cannot " "exceed number of layers."
                )
            if ix >= (num_blocks - num_se_blocks):
                use_se = True

            # Depthwise conv
            blocks.append(
                MobileOneBlock(
                    in_channels=self.in_planes,
                    out_channels=self.in_planes,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=self.in_planes,
                    inference_mode=self.inference_mode,
                    use_se=use_se,
                    num_conv_branches=self.num_conv_branches,
                )
            )
            # Pointwise conv
            blocks.append(
                MobileOneBlock(
                    in_channels=self.in_planes,
                    out_channels=planes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                    inference_mode=self.inference_mode,
                    use_se=use_se,
                    num_conv_branches=self.num_conv_branches,
                )
            )
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


PARAMS = {
    "s0": {"width_multipliers": (0.75, 1.0, 1.0, 2.0), "num_conv_branches": 4},
    "s1": {"width_multipliers": (1.5, 1.5, 2.0, 2.5)},
    "s2": {"width_multipliers": (1.5, 2.0, 2.5, 4.0)},
    "s3": {"width_multipliers": (2.0, 2.5, 3.0, 4.0)},
    "s4": {"width_multipliers": (3.0, 3.5, 3.5, 4.0), "use_se": True},
}


def mobileone(
    num_classes: int = 1000, inference_mode: bool = False, variant: str = "s0"
) -> nn.Module:
    """Get MobileOne model.

    :param num_classes: Number of classes in the dataset.
    :param inference_mode: If True, instantiates model in inference mode.
    :param variant: Which type of model to generate.
    :return: MobileOne model."""
    variant_params = PARAMS[variant]
    return MobileOne(
        num_classes=num_classes, inference_mode=inference_mode, **variant_params
    )
