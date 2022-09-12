import numpy as np

import torch
from torch.nn import functional as F
from mmocr.utils.ocr import MMOCR


class MMOCRInferCRNN:
    def __init__(self, rule_func=None, return_img: bool = False):
        self.return_img = return_img
        self.model = MMOCR(det=None, recog="CRNN")

        if rule_func is None:

            def rule_func(pred):
                return pred

        self.rules = rule_func

    def denormalize(self, x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        ten = x.clone()
        for t, m, s in zip(ten, mean, std):
            t.mul_(s).add_(m)
        return torch.clamp(ten, 0, 1).mul_(255)

    def __call__(self, char_pos, image, img_size: tuple = (16, 16)):

        pred_char_list = []
        pred_img_char_list = []

        image = self.denormalize(image[0])[
            None,
        ]

        for i in char_pos:
            _max, _min = i
            _image = image[:, :, _min[1] : _max[1], _min[0] : _max[0]]
            _image = F.pad(_image, (2, 2, 2, 2), value=255)
            _image = F.interpolate(_image, size=img_size, mode="bilinear")
            # _image = F.interpolate(_image, size=img_size)
            _image = np.array(_image[0].permute(1, 2, 0), dtype=np.uint8)

            pred = self.model.single_inference(
                self.model.recog_model, [_image], False, 0
            )[0]["text"]
            pred = self.rules(pred)

            pred_char_list.append(pred)
            if self.return_img:
                pred_img_char_list.append(_image)

        return pred_char_list, pred_img_char_list
