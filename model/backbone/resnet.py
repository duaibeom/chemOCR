from torch import nn


class ResNet(nn.Module):

    def __init__(self, size: int, color_BW: bool = True) -> None:
        super().__init__()

        if size == 18:
            from torchvision.models import resnet18
            self.model = resnet18()
            self.in_channels_list = [64, 128, 256, 512],
        elif size == 50:
            from torchvision.models import resnet50
            self.model = resnet50()
            self.in_channels_list = [256, 512, 1024, 2048],
        else:
            raise ValueError

        self.return_layers = dict(
            layer1='feature1',
            layer2='feature2',
            layer3='feature3',
            layer4='feature4',
        )

        if color_BW:
            self.model.conv1 = nn.Conv2d(1,
                                         64,
                                         kernel_size=(7, 7),
                                         stride=(2, 2),
                                         padding=(3, 3),
                                         bias=False)
