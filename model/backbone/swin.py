from torch import nn


class SwinTransformer(nn.Module):

    def __init__(self,
                 size: str,
                 color_BW: bool = True,
                 pretrained: bool = False) -> None:
        super().__init__()

        if size == 't':
            if pretrained:
                from torchvision.models import swin_t, Swin_T_Weights
                self.model = swin_t(weights=Swin_T_Weights).features
            else:
                from torchvision.models import swin_t
                self.model = swin_t().features
            self.in_channels_list = [96, 192, 384, 768]
        elif size == 's':
            if pretrained:
                from torchvision.models import swin_s, Swin_S_Weights
                self.model = swin_s(weights=Swin_S_Weights).features
            else:
                from torchvision.models import swin_s
                self.model = swin_s().features
            self.in_channels_list = [96, 192, 384, 768]
        elif size == 'b':
            if pretrained:
                from torchvision.models import swin_b, Swin_B_Weights
                self.model = swin_b(weights=Swin_B_Weights).features
            else:
                from torchvision.models import swin_b
                self.model = swin_b().features
            self.in_channels_list = [128, 256, 512, 1024]
        else:
            raise ValueError

        self.return_layers = {
            '1': 'feature1',
            '3': 'feature2',
            '5': 'feature3',
            '7': 'feature4',
        }

        if color_BW:
            _conv_o_ch = self.model[0][0].weight.shape[0]
            self.model[0][0] = nn.Conv2d(
                1,
                _conv_o_ch,
                kernel_size=4,
                stride=(4, 4),
            )
