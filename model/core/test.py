# built-in
import time
import logging

# 3rd
import numpy as np
from tqdm import tqdm

# torch
import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast

from utils import label_accuracy_score, add_hist
from datasets import CLASSES

logger = logging.getLogger('train')


def test(model,
         dataloaders,
         optimizer,
         criterion,
         device,
         autocast_enabled: bool = False,
         phase: str = 'test'):

    model.eval()  # Set model to evaluate mode

    n_iter = 0
    cost_time = 0
    eval_time = 0
    hist_time = 0
    epoch_loss = 0

    hist = np.zeros((22, 22))
    for images, masks, image_infos in tqdm(dataloaders[phase]):
        n_iter += 1
        optimizer.zero_grad()

        images = torch.stack(images).to(device)
        masks = torch.stack(masks).long().to(device)

        _time = time.perf_counter()
        outputs = model.forward(images)['out']
        cost_time += time.perf_counter() - _time

        _time = time.perf_counter()
        masks = masks.cpu().numpy()
        outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()

        hist = add_hist(hist, masks, outputs, n_class=22)
        hist_time += time.perf_counter() - _time

    _time = time.perf_counter()
    acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
    print_metric(IoU)
    eval_time += time.perf_counter() - _time
    logger.info(
        f"{phase.upper():5} TIME: Cost - {cost_time:.2f}s, Hist - {hist_time:.2f}s, Eval - {eval_time:.2f}s"
    )


def print_metric(values):
    _metric_by_class = {
        class_name: round(IoU, 4)
        for class_name, IoU in zip(CLASSES, values)
    }
    logger.info(f'IoU by class :')
    max_row = 4
    n_row = 0
    template = ''
    for key, _values in _metric_by_class.items():
        template += f"| {key:28} | {_values:.4f} |"
        n_row += 1
        if max_row == n_row:
            logger.info(template)
            template = ''
            n_row = 0
    logger.info(template)
