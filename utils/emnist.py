import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

eminst_class = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",  # index 61
    "-",  # index 61
    "+",  # index 61
    "Br",
    "Cl",
]


# def denormalize(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
def denormalize(x, mean=[0.5], std=[0.22]):
    # 3, H, W, B
    ten = x.clone()
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    ten = ten.permute(1, 2, 0)
    return torch.clamp(ten, 0, 1).detach().numpy()


def get_train_transform(image):
    image = np.array(image, dtype=np.int16)
    image -= 255
    image *= -1
    image = image.astype(np.uint8)[:, :, None].repeat(3, axis=2)
    x = train_transform(image=image)
    return x["image"]


def get_test_transform(image):
    image = np.array(image, dtype=np.int16)
    image -= 255
    image *= -1
    image = image.astype(np.uint8)[:, :, None].repeat(3, axis=2)
    x = test_transform(image=image)
    return x["image"]


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 2, 2),
            nn.Conv2d(256, 256, 2, 1),
            nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        # self.fc = nn.Linear(256, 62)
        self.fc = nn.Conv2d(256, 512, 1)
        self.fc1 = nn.Conv2d(512, 100, 1)

    def forward(self, x):
        x = self.conv_stack(x)
        # x = x.view(256, -1)
        x = self.fc(x)
        x = self.fc1(x)
        x = x.squeeze(-1)
        x = x.squeeze(-1)
        return x


class PredictAtomChar:
    def __init__(self, rule_func=None, return_img: bool = False):
        self.return_img = return_img
        self.model = NeuralNetwork()
        self.model.load_state_dict(torch.load("utils/model_weights.emnist.pth"))
        # self.model.load_state_dict(torch.load("utils/model_weights.fix.pth"))
        self.model.eval()

        if rule_func is None:

            def rule_func(pred):
                return pred

        self.rules = rule_func

    def denormalize(self, x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        ten = x.clone()
        for t, m, s in zip(ten, mean, std):
            t.mul_(s).add_(m)
        return torch.clamp(ten, 0, 1)

    def pre_img(self, _img):
        # _img = self.denormalize(_img)
        # x = (
        #     0.2126 * _img[:, 2, :, :]
        #     + 0.7152 * _img[:, 1, :, :]
        #     + 0.0722 * _img[:, 0, :, :]
        # )
        x = 0.33 * _img[:, 0, :, :] + 0.33 * _img[:, 1, :, :] + 0.33 * _img[:, 2, :, :]
        # x = (x - 0.5) / 0.22
        # x = ((x > 0.8).to(dtype=torch.float32) - 0.5) / 0.22
        return x[
            :,
            None,
        ]

    def __call__(self, char_pos, image, img_size: tuple = (14, 14)):

        pred_char_list = []
        pred_img_char_list = []

        image = self.pre_img(image)

        for i in char_pos:
            _max, _min = i
            _image = image[:, :, _min[1] : _max[1], _min[0] : _max[0]]
            _image = F.pad(_image, (1, 1, 1, 1), value=2.2727)
            # _image = F.interpolate(_image, size=img_size, mode="bilinear")
            _image = F.interpolate(_image, size=img_size)
            pred = eminst_class[self.model(_image).argmax()]
            # pred = "0"
            pred = self.rules(pred)

            pred_char_list.append(pred)
            if self.return_img:
                pred_img_char_list.append(denormalize(_image[0]))

        if self.return_img:
            return pred_char_list, pred_img_char_list
        else:
            return pred_char_list


def train(dataloader, model, loss_fn, optimizer, autocast_enabled):

    if autocast_enabled:
        scaler = GradScaler()

    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        optimizer.zero_grad()
        with torch.set_grad_enabled(True), autocast(enabled=autocast_enabled):
            pred = model(X)
            loss = loss_fn(pred, y)

            if not autocast_enabled:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


if __name__ == "__main__":

    from torch.utils.data import Dataset
    from torchvision import datasets

    from torch.utils.data import DataLoader

    from torch.cuda.amp.grad_scaler import GradScaler
    from torch.cuda.amp.autocast_mode import autocast

    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    train_transform = A.Compose(
        [
            A.Resize(20, 20),
            # A.InvertImg(p=1),
            A.HorizontalFlip(True),
            A.Rotate((90, 90), always_apply=True),
            # A.Blur(blur_limit=3, p=0.4),
            A.RandomBrightnessContrast(p=0.4),
            # A.GaussNoise(p=0.3),
            # A.CoarseDropout(4, 2, 2, p=0.3),
            # A.Normalize(mean=(0.5), std=(0.22)),
            A.Normalize(),
            ToTensorV2(),
        ]
    )
    test_transform = A.Compose(
        [
            A.Resize(20, 20),
            # A.InvertImg(p=1),
            A.HorizontalFlip(True),
            A.Rotate((90, 90), always_apply=True),
            A.Normalize(mean=(0.5), std=(0.22)),
            # A.Normalize(),
            ToTensorV2(),
        ]
    )

    training_data = datasets.EMNIST(
        split="byclass",
        # split='letters',
        root="emnist_data",
        train=True,
        download=True,
        transform=get_train_transform,
    )

    test_data = datasets.EMNIST(
        split="byclass",
        # split='letters',
        root="emnist_data",
        train=False,
        download=True,
        transform=get_test_transform,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = NeuralNetwork()
    # model.load_state_dict(torch.load("model_weights.emnist.pth"), strict=False)
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    train_dataloader = DataLoader(
        training_data, batch_size=1024, shuffle=True, num_workers=4
    )
    test_dataloader = DataLoader(
        test_data, batch_size=1024, shuffle=True, num_workers=4
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, True)
        test(test_dataloader, model, loss_fn)
        # if t % 5 == 0:
    print("Done!")

    torch.save(model.state_dict(), "model_weights.emnist.pth")
