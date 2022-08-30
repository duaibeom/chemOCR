import time
import random
import functools

import numpy as np
from matplotlib import pyplot as plt

import torch


def timer(func):

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value

    return wrapper_timer


def set_random_seed(random_seed,
                    multi_gpu: bool = False,
                    deterministic: bool = True,
                    cudnn_benchmark: bool = False):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    if multi_gpu:
        torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark
    np.random.seed(random_seed)
    random.seed(random_seed)


class MetricList:

    def __init__(self,
                 train_process: list[str],
                 figname: str,
                 nrows: int,
                 ncols: int,
                 dpi: int = 150) -> None:
        self.phase = train_process
        self.dict = {process: dict() for process in train_process}
        self.nrows = nrows
        self.ncols = ncols
        self.figsize = (6 * self.ncols, 2 + self.nrows * 4)
        self.dpi = dpi
        self.figname = figname

    def update(self, metric):
        for process in metric:
            if process not in self.phase:
                raise ValueError

            _metric = metric[process]

            for k in _metric:
                try:
                    self.dict[process][k].append(_metric[k])
                except KeyError:
                    self.dict[process][k] = [_metric[k]]

    def plot(self):
        fig, ax = plt.subplots(self.nrows,
                               self.ncols,
                               figsize=self.figsize,
                               dpi=100)

        q, w, e = 0, 0, 0
        linestyle = ''

        for process in self.phase:
            q = 0
            linestyle += '-'
            for k, v in self.dict[process].items():
                if self.nrows > 1:
                    pos = (w, q)
                else:
                    pos = q
                ax[pos].plot(v, linestyle=linestyle, label=f'{process}_{k}')
                ax[pos].legend()
                q += 1
                if q == self.ncols:
                    w += 1

        fig.tight_layout()
        fig.savefig(f"{self.figname}.png")
        plt.cla()
        plt.clf()
        del fig, ax

    def wandb_update(self, wandb):
        wandb.log({
            f"{process}/{k}": v[-1]
            for process in self.phase for k, v in self.dict[process].items()
        })


# def print_metric(values):
#     _metric_by_class = {
#         class_name: round(IoU, 4)
#         for class_name, IoU in zip(CLASSES, values)
#     }
#     logger.info(f'IoU by class :')
#     max_row = 4
#     n_row = 0
#     template = ''
#     for key, _values in _metric_by_class.items():
#         template += f"| {key:28} | {_values:.4f} |"
#         n_row += 1
#         if max_row == n_row:
#             logger.info(template)
#             template = ''
#             n_row = 0
#     logger.info(template)