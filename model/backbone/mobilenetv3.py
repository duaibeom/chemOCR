from torch import nn


class MobileNetV3(nn.Module):

    def __init__(self,
                 size: str,
                 reduced_tail: bool = True,
                 color_BW: bool = True) -> None:
        super().__init__()
        if size == 'large':
            from torchvision.models import mobilenet_v3_large
            self.model = mobilenet_v3_large(reduced_tail=reduced_tail).features
            self.return_layers = {
                '2': 'feature1',
                '4': 'feature2',
                '7': 'feature3',
                '13': 'feature4',
                '16': 'feature5',
            }
            self.in_channels_list = [24, 40, 80, 80, 480]

        elif size == 'small':
            from torchvision.models import mobilenet_v3_small
            self.model = mobilenet_v3_small(reduced_tail=reduced_tail).features
            self.return_layers = {
                '1': 'feature1',
                '2': 'feature2',
                '4': 'feature3',
                '9': 'feature4',
                '12': 'feature5',
            }
            self.in_channels_list = [16, 24, 40, 48, 288]

        else:
            raise ValueError

        if color_BW:
            self.model[0][0] = nn.Conv2d(1,
                                         16,
                                         kernel_size=3,
                                         stride=(2, 2),
                                         padding=1,
                                         bias=False)
