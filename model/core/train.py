# built-in
import time
import logging

# 3rd
# import numpy as np
from tqdm import tqdm

# torch
import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast

logger = logging.getLogger("train")


def train_one_epoch(
    epoch: int,
    model: torch.nn.Module,
    dataloaders,
    optimizer,
    criterion,
    device,
    scheduler=None,
    train_process: list = ["train", "val"],
    autocast_enabled: bool = False,
):

    if autocast_enabled:
        scaler = GradScaler()

    metrics = {}

    for phase in train_process:

        if phase == "train":
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        n_iter = 0
        cost_time = 0
        eval_time = 0
        hist_time = 0
        total_loss = 0
        total_prob_loss = 0
        total_bi_loss = 0
        total_thr_loss = 0
        total_cat_loss = 0

        # hist = np.zeros((22, 22))
        for images, gt_shr, gt_shr_mask, gt_thr, gt_thr_mask in tqdm(
            dataloaders[phase]
        ):
            n_iter += 1
            optimizer.zero_grad()

            images = torch.stack(images).to(device)
            gt_shr = torch.stack(gt_shr).long().to(device)
            gt_shr_mask = torch.stack(gt_shr_mask).to(device)
            # gt_shr_mask = torch.ones_like(gt_shr).to(device)
            gt_thr = torch.stack(gt_thr).to(device)
            gt_thr_mask = torch.stack(gt_thr_mask).to(device)
            # gt_thr_mask = torch.ones_like(gt_thr).to(device)

            _time = time.perf_counter()
            with torch.set_grad_enabled(phase == "train"), autocast(
                enabled=autocast_enabled
            ):
                outputs = model(images)
                losses = criterion(outputs, gt_shr, gt_shr_mask, gt_thr, gt_thr_mask)

                loss = (
                    losses["loss_prob"]
                    + losses["loss_bi"]
                    + losses["loss_thr"]
                    + losses["loss_cat"]
                )
                # loss = losses["loss_prob"] + losses["loss_bi"] + losses["loss_thr"]

                if phase == "train":
                    if not autocast_enabled:
                        loss.backward()
                        optimizer.step()
                    else:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

            total_loss += loss.cpu().item()
            total_prob_loss += losses["loss_prob"].cpu().item()
            total_bi_loss += losses["loss_bi"].cpu().item()
            total_thr_loss += losses["loss_thr"].cpu().item()
            total_cat_loss += losses["loss_cat"].cpu().item()

            cost_time += time.perf_counter() - _time

            _time = time.perf_counter()

        #
        total_loss /= n_iter
        total_prob_loss /= n_iter
        total_thr_loss /= n_iter
        total_thr_loss /= criterion.beta
        total_bi_loss /= n_iter
        total_bi_loss /= criterion.alpha
        total_cat_loss /= n_iter
        total_cat_loss /= criterion.gamma

        _time = time.perf_counter()
        logger.info(
            f"{phase.upper():5}: Epoch [{epoch+1}], Loss: {round(total_loss, 4)}, Prob: {round(total_prob_loss, 4)}, Bi: {round(total_bi_loss, 4)}, THR: {round(total_thr_loss, 4)}, CAT: {round(total_cat_loss, 4)}"
        )
        # print_metric(IoU)
        eval_time += time.perf_counter() - _time
        logger.info(
            f"{phase.upper():5} TIME: Cost - {cost_time:.2f}s, Hist - {hist_time:.2f}s, Eval - {eval_time:.2f}s"
        )

        metrics[phase] = dict(
            loss=total_loss,
            prob_loss=total_prob_loss,
            bi_loss=total_bi_loss,
            thr_loss=total_thr_loss,
            cat_loss=total_cat_loss,
        )

    return metrics
