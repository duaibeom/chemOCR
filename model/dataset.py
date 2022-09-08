# import os

import numpy as np
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset
from model.utils.utils import timer
from model.utils.transfroms import CustomRandomPadImage, CustomCenterPadImage

custom_rnd_aug = CustomRandomPadImage(512, 512)
custom_cnt_aug = CustomCenterPadImage(50)


@timer
class CustomDataset(Dataset):
    def __init__(self, data_df: str, mode: str, dir_path: str, transforms=None):
        super().__init__()
        self.mode = mode
        self.df = pd.read_csv(data_df)
        self.dir_path = dir_path
        self.transforms = transforms

    def load_image(self, file_name: str, mask: bool = False):
        with Image.open(file_name) as pikybow_image:
            image = np.array(pikybow_image, dtype=np.uint8)
        if not mask:
            # image = image.astype(np.float32)
            # image /= 255
            image = image[:, :, :3]
        return image

    def __getitem__(
        self,
        index: int,
    ):

        _id = self.df.iloc[index, 0]

        rnd_int = np.random.randint(0, 4)

        if self.mode in ("train", "val"):
            _dir = f"{self.dir_path}_{rnd_int}/{_id[6]}/{_id[7]}/{_id[8]}/{_id[9]}/{_id[10]}"
        else:
            _dir = f"{self.dir_path}/{_id[6]}/{_id[7]}/{_id[8]}/{_id[9]}/{_id[10]}"

        image = self.load_image(f"{_dir}/{_id}.png")

        if self.mode in ("train", "val"):
            gt_shr = self.load_image(f"{_dir}/{_id}.shr.png", mask=True)
            # gt_shr_mask = self.load_image(f"{_dir}/{_id}.shr_mask.png", mask=True)
            gt_shr_mask = (gt_shr > 0).astype(np.uint8)
            gt_thr = (
                self.load_image(f"{_dir}/{_id}.thr.png", mask=True).astype(np.float32)
                / 255
            )
            gt_thr_mask = self.load_image(f"{_dir}/{_id}.thr_mask.png", mask=True)

            masks = [gt_shr, gt_shr_mask, gt_thr, gt_thr_mask]

            image, masks = custom_rnd_aug(image=image, masks=masks)

            # transform -> albumentations 라이브러리 활용
            if self.transforms is not None:

                transformed = self.transforms(image=image, masks=masks)
                image = transformed["image"]
                masks = transformed["masks"]

            return image, masks[0], masks[1], masks[2], masks[3]

        elif self.mode == "test":

            image = custom_cnt_aug(image=image)

            # transform -> albumentations 라이브러리 활용
            if self.transforms is not None:
                transformed = self.transforms(image=image)
                image = transformed["image"]
            return image  # , image_infos

    def __len__(self) -> int:
        return len(self.df)
