import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2


class CustomCenterImage:
    def __init__(self, min_height: int = 224, min_width: int = 224) -> None:
        self.max_h = min_height
        self.max_w = min_width

    def __call__(self, image, masks=None):
        raise PermissionError
        h, w = image.shape

        max_w = self.max_w
        max_h = self.max_h

        gap_w = max_w - w
        gap_h = max_h - h

        max_w = w + self.pad
        max_h = h + self.pad

        pad_w = self.pad // 2
        pad_h = pad_w

        image_bg = np.ones((max_h, max_w), dtype=image.dtype)

        image_bg[pad_h : pad_h + h, pad_w : pad_w + w] = image

        if masks:
            _masks = []
            for mask in masks:
                _bg = np.zeros((max_h, max_w), dtype=mask.dtype)
                # if mask.dtype == np.float32:
                #     _bg += 0.3
                _bg[pad_h : pad_h + h, pad_w : pad_w + w] = mask
                _masks.append(_bg)

            return image_bg, _masks

        return image_bg


class CustomCenterPadImage:
    def __init__(self, pad_size: int = 10) -> None:
        self.pad = pad_size

    def __call__(self, image, masks=None):
        h, w = image.shape[:2]
        # h, w = image.shape

        # max_length = max(h, w)

        max_w = w + self.pad
        max_h = h + self.pad

        pad_w = self.pad // 2
        pad_h = pad_w

        if image.shape.__len__() >= 3:
            image_bg = np.ones((max_h, max_w, 3), dtype=image.dtype)
            image_bg[pad_h : pad_h + h, pad_w : pad_w + w, :] = image
        else:
            image_bg = np.ones((max_h, max_w), dtype=image.dtype)
            image_bg[pad_h : pad_h + h, pad_w : pad_w + w] = image

        if masks:
            _masks = []
            for mask in masks:
                _bg = np.zeros((max_h, max_w), dtype=mask.dtype)
                # if mask.dtype == np.float32:
                #     _bg += 0.3
                _bg[pad_h : pad_h + h, pad_w : pad_w + w] = mask
                _masks.append(_bg)

            return image_bg, _masks

        return image_bg


class CustomRandomPadImage:
    def __init__(self, min_height: int = 224, min_width: int = 224) -> None:
        self.max_h = min_height
        self.max_w = min_width

    def __call__(self, image, masks=None):
        h, w = image.shape[:2]

        max_w = self.max_w
        max_h = self.max_h

        gap_w = max_w - w
        gap_h = max_h - h

        rnd_w = 0
        rnd_h = 0

        if (gap_w < 0) or (gap_h < 0):

            if h >= w:
                max_w = h
                max_h = h
            # elif h < w:
            else:
                max_w = w
                max_h = w

            gap_w = max_w - w
            gap_h = max_h - h

        if gap_w > 0:
            rnd_w = np.random.randint(0, gap_w)
        if gap_h > 0:
            rnd_h = np.random.randint(0, gap_h)

        if image.shape.__len__() >= 3:
            image_bg = np.ones((max_h, max_w, 3), dtype=image.dtype)
            image_bg[rnd_h : rnd_h + h, rnd_w : rnd_w + w, :] = image
        else:
            image_bg = np.ones((max_h, max_w), dtype=image.dtype)
            image_bg[rnd_h : rnd_h + h, rnd_w : rnd_w + w] = image

        if masks:
            _masks = []
            for mask in masks:
                _bg = np.zeros((max_h, max_w), dtype=mask.dtype)
                # if mask.dtype == np.float32:
                #     _bg += 0.3
                _bg[rnd_h : rnd_h + h, rnd_w : rnd_w + w] = mask
                _masks.append(_bg)

            return image_bg, _masks

        return image_bg


def get_train_transform():
    return A.Compose(
        [
            # A.PadIfNeeded(512, 512),
            A.Resize(512, 512),
            A.ShiftScaleRotate(
                scale_limit=(-0.3, 0.5), rotate_limit=0, shift_limit=0, p=0.5
            ),
            # A.HorizontalFlip(p=0.5),
            A.ChannelShuffle(p=0.4),
            A.Blur(blur_limit=5, p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(var_limit=(0, 255), p=0.3),
            # A.VerticalFlip(p=0.4),
            # A.InvertImg(p=0.3),
            # A.Normalize(mean=(0.5), std=(0.22), max_pixel_value=1),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


def get_valid_transform():
    return A.Compose(
        [
            # A.Resize(240, 427),
            A.Resize(512, 512),
            # A.RandomCrop(240, 320),
            # A.Normalize(mean=(0.5), std=(0.22), max_pixel_value=1),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


def get_test_transform():
    return A.Compose(
        [
            # A.Normalize(mean=(0.5), std=(0.22), max_pixel_value=1)
            A.Normalize(),
            ToTensorV2(),
        ]
    )
