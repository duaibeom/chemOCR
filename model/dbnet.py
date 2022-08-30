# import torch
from torch import nn, Tensor

from .head.db_head import DBHead, DBHeadTEST
from .neck.fpn import BackboneWithFPNC

from .backbone.mobilenetv3 import MobileNetV3

# from .backbone.swin import SwinTransformer


class DBNet(nn.Module):
    def __init__(
        self,
        inner_channels: int,
        out_channels: int,
        head_in_channels: int,
        permute: bool = False,
        test: bool = False,
    ) -> None:
        super().__init__()
        backbone = MobileNetV3(size="small")
        # backbone = SwinTransformer(size='t')
        self.backbone = backbone.model
        self.neck = BackboneWithFPNC(
            backbone=self.backbone,
            return_layers=backbone.return_layers,
            in_channels_list=backbone.in_channels_list,
            inner_channels=inner_channels,
            out_channels=out_channels,
            permute=permute,
        )
        if test:
            self.head = DBHeadTEST(in_channels=head_in_channels)
        else:
            self.head = DBHead(in_channels=head_in_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.neck(x)
        x = self.head(x)
        return x
