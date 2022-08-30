from typing import Callable

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from torchvision.models._utils import IntermediateLayerGetter
# from torchvision.ops.feature_pyramid_network import ExtraFPNBlock, FeaturePyramidNetwork, LastLevelMaxPool


class BackboneWithFPNC(nn.Module):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediateLayerGetter apply here.
    Args:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
        norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
    Attributes:
        out_channels (int): the number of channels in the FPN
    """

    def __init__(
        self,
        backbone: nn.Module,
        return_layers: dict[str, str],
        in_channels_list: list[int],
        inner_channels: int,
        out_channels: int,
        permute: bool = False,
        # extra_blocks: ExtraFPNBlock | None = None,
        # norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()

        # if extra_blocks is None:
        #     extra_blocks = LastLevelMaxPool()

        self.body = IntermediateLayerGetter(backbone,
                                            return_layers=return_layers)
        # self.fpn = FeaturePyramidNetwork(
        #     in_channels_list=in_channels_list,
        #     out_channels=out_channels,
        #     extra_blocks=extra_blocks,
        #     norm_layer=norm_layer,
        # )

        self.lateral_conv = nn.ModuleList([
            nn.Conv2d(in_channel, inner_channels, 1)
            for in_channel in in_channels_list
        ])
        self.smooth_conv = nn.ModuleList([
            nn.Conv2d(inner_channels, out_channels, 3, padding=1)
            for _ in in_channels_list
        ])

        self.out_channels = out_channels
        self.permute = permute

    def forward(self, x: Tensor) -> dict[str, Tensor]:

        x = self.body(x)
        # names = list(x.keys())
        x = list(x.values())
        if self.permute:
            x = [_x.permute(0, 3, 1, 2) for _x in x]
        # x = self.fpn(x)

        x = [self.lateral_conv[i](_feat) for i, _feat in enumerate(x)]

        first_inner_size = x[0].shape[-2:]
        last_inner = x[-1]
        results = []
        results.append(last_inner)

        for i in range(len(x) - 2, -1, -1):
            inner_lateral = x[i]
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner,
                                           size=feat_shape,
                                           mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, last_inner)

        x = [self.smooth_conv[i](_feat) for i, _feat in enumerate(results)]

        for i, _feat in enumerate(x[1:]):
            x[i + 1] = F.interpolate(_feat, size=first_inner_size)

        x = torch.cat(x, 1)

        return x