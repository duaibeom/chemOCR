## built-in
import time
import logging
from datetime import datetime

## 3rd
import torch
from torch import optim
from torch.utils.data import DataLoader

## Custom
from model.dbnet import DBNet
from model.loss import DBLoss
from model.dataset import CustomDataset

from model.core.train import train_one_epoch

from model.utils.transfroms import get_train_transform, get_valid_transform
from model.utils.utils import set_random_seed, MetricList

now = datetime.now()
cur_time_str = now.strftime("%d%m%Y_%H%M")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-8s %(levelname)-6s %(message)s",
    datefmt="%m-%d %H:%M",
    filename=f"log/{cur_time_str}.log",
    filemode="w",
)

logger = logging.getLogger("main")


def collate_fn(batch):
    return tuple(zip(*batch))


def main(args):

    # ------ CONFIG
    HOME = ".."
    DATA_DIR = f"{HOME}/Data/ChEMBL/OCR_RGB"
    TRAIN_PROCESS = ["train", "val"]
    DATAFRAME_LIST = dict(
        train=f"data/chembl_31_smiles_train.csv",
        val=f"data/chembl_31_smiles_val.csv",
        test=f"data/chembl_31_smiles_test.csv",
    )

    # ------ WANDB CONFIG
    # wandb.config = {**args}

    # ------ RANDOM SEED
    set_random_seed(123456)

    # ------ TRANSFORM
    # TODO
    defined_transforms = {
        "train": get_train_transform(),
        "val": get_valid_transform(),
    }

    # ------ DATASET
    ocr_dataset = {
        x: CustomDataset(
            data_df=DATAFRAME_LIST[x],
            transforms=defined_transforms[x],
            mode=x,
            dir_path=DATA_DIR,
        )
        for x in TRAIN_PROCESS
    }

    # ------ DATALOADER
    _time = time.perf_counter()
    dataloaders = {
        x: DataLoader(
            ocr_dataset[x],
            batch_size=args["batch_size"],
            shuffle=True,
            num_workers=3,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        for x in TRAIN_PROCESS
    }
    logger.info(f"Dataloader progress. {time.perf_counter() - _time:.4f}s")

    # ------ MODEL
    _time = time.perf_counter()
    model = DBNet(
        inner_channels=192,
        out_channels=96,
        head_in_channels=480,
    )
    # model.load_state_dict(
    #     torch.load("backup/model_weights.mbv3s.5n128h320.8c.pth"), strict=False
    # )
    # model.load_state_dict(torch.load("model_weights.v9.mbv3s.final.pth"), strict=False)
    model.load_state_dict(torch.load("model_weights.v9_rgb.mbv3s.5n192h480.final.pth"))
    # model.load_state_dict(torch.load("backup/model_weights.mbv3s.5n128h320.8c.pth"))
    model.to(device)
    logger.info(f"Model progress. {time.perf_counter() - _time:.4f}s")

    # ------ OPTIMIZER
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args["lr"],
    )

    # ------ LOSS
    _wy = torch.Tensor([0.3, 0.2, 0.4, 0.2, 0.2, 1, 0.6, 1]).to(device)
    loss_func = DBLoss(
        alpha=1,
        beta=10,
        gamma=0.4,
        negative_ratio=3,
        # downscaled=True,
        ce_weight=_wy,
    )

    # ------ TRAINING
    N_EPOCH = args["epochs"]

    metric = MetricList(TRAIN_PROCESS, cur_time_str, 1, 5)

    for epoch in range(N_EPOCH):
        logger.info(f"Epoch {epoch + 1:>3}/{N_EPOCH} ----------")
        metrics = train_one_epoch(
            epoch=epoch,
            model=model,
            dataloaders=dataloaders,
            optimizer=optimizer,
            device=device,
            criterion=loss_func,
            train_process=TRAIN_PROCESS,
            autocast_enabled=args["fp16"],
        )

        metric.update(metrics)
        metric.plot(save_path="log")
        # metric.wandb_update()

        if (epoch + 1) % 5 == 0:
            torch.save(
                model.state_dict(),
                f"model_weights.v9_rgb.mbv3s.5n192h480.{epoch + 1}.pth",
            )

    torch.save(model.state_dict(), "model_weights.v9_rgb.mbv3s.5n192h480.final.pth")


if __name__ == "__main__":

    # wandb.init(project="", entity="")

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    params = dict(epochs=60, lr=5e-5, batch_size=32, fp16=True)
    main(params)
