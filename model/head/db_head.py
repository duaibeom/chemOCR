import torch
from torch import nn, Tensor
from torch.nn import functional as F


class DBHead(nn.Module):
    """The class for DBNet head.
    Args:
        in_channels (int): The number of input channels of the db head.
        with_bias (bool): Whether add bias in Conv2d layer.
        downsample_ratio (float): The downsample ratio of ground truths.

    reference:
        1. MMOCR - DBNet
        2. https://github.com/MhLiao/DB
    """

    def __init__(
        self,
        in_channels: int,
        with_bias: bool = False,
        downsample_ratio: float = 1.0,
        **kwargs
    ):
        super().__init__()

        assert isinstance(in_channels, int)

        def _init_binarize(in_channels: int, bias: bool = with_bias):
            # _inner_channels = in_channels // 4
            _inner_channels = int(in_channels * 0.25)
            return nn.Sequential(
                nn.Conv2d(in_channels, _inner_channels, 3, padding=1, bias=bias),
                nn.BatchNorm2d(_inner_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(_inner_channels, _inner_channels, 2, 2),
                nn.BatchNorm2d(_inner_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(_inner_channels, 1, 2, 2),
                nn.Sigmoid(),
            )

        def _init_categorize(
            in_channels: int,
            out_channels: int,
            in_channel_scale: int = 1,
            bias: bool = with_bias,
        ):
            _inner_channels = int(in_channels * in_channel_scale)
            return nn.Sequential(
                nn.Conv2d(in_channels, _inner_channels, 3, padding=1, bias=bias),
                nn.BatchNorm2d(_inner_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(_inner_channels, _inner_channels, 2, 2),
                nn.BatchNorm2d(_inner_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(_inner_channels, out_channels, 2, 2),
                nn.Softmax(dim=1),
            )

        self.in_channels = in_channels
        self.downsample_ratio = downsample_ratio

        # self.binarize = _init_binarize(in_channels)
        self.categorize = _init_categorize(
            in_channels, out_channels=8, in_channel_scale=1.0
        )
        self.threshold = _init_binarize(in_channels)

    def diff_appx_binarize(self, prob_map, thr_map, k: int = 50):
        return torch.reciprocal(1.0 + torch.exp(-k * (prob_map - thr_map)))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            inputs (Tensor): Shape (batch_size, hidden_size, h, w).
        Returns:
            Tensor: A tensor of the same shape as input.
        """
        # prob_map = self.binarize(x)

        prob_cat_map = self.categorize(x)
        prob_map = prob_cat_map[:, 1:].sum(dim=1, keepdim=True)

        thr_map = self.threshold(x)
        binary_map = self.diff_appx_binarize(prob_map, thr_map, k=50)

        outputs = torch.cat((prob_map, thr_map, binary_map, prob_cat_map), dim=1)
        # outputs = torch.cat((prob_map, thr_map, binary_map), dim=1)
        return outputs


class DBHeadTEST(DBHead):
    """The class for DBNet head.
    Args:
        in_channels (int): The number of input channels of the db head.
        with_bias (bool): Whether add bias in Conv2d layer.
        downsample_ratio (float): The downsample ratio of ground truths.

    reference:
        1. MMOCR - DBNet
        2. https://github.com/MhLiao/DB
    """

    def __init__(
        self,
        in_channels: int,
        with_bias: bool = False,
        downsample_ratio: float = 1.0,
        **kwargs
    ):
        super().__init__(
            in_channels,
            with_bias=with_bias,
            downsample_ratio=downsample_ratio,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            inputs (Tensor): Shape (batch_size, hidden_size, h, w).
        Returns:
            Tensor: A tensor of the same shape as input.
        """
        return self.categorize(x)
