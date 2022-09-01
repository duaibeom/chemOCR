import numpy as np

import torch
from torch import nn


def get_train_transform(image):
    image = np.array(image, dtype=np.uint8)
    x = train_transform(image=image)
    return x["image"]


def get_test_transform(image):
    image = np.array(image, dtype=np.uint8)
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
        self.fc1 = nn.Conv2d(512, 62, 1)

    def forward(self, x):
        x = self.conv_stack(x)
        # x = x.view(256, -1)
        x = self.fc(x)
        x = self.fc1(x)
        x = x.squeeze(-1)
        x = x.squeeze(-1)
        return x


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
            A.Resize(14, 14),
            A.HorizontalFlip(True),
            A.Rotate((90, 90), always_apply=True),
            # A.Blur(blur_limit=4, p=0.3),
            # A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(var_limit=(0, 0.2), p=0.3),
            # A.CoarseDropout(4, 3, p=0.3),
            A.InvertImg(p=1),
            A.Normalize(mean=(0.45), std=(0.22), max_pixel_value=255),
            ToTensorV2(),
        ]
    )
    test_transform = A.Compose(
        [
            A.Resize(14, 14),
            A.HorizontalFlip(True),
            A.Rotate((90, 90), always_apply=True),
            A.InvertImg(p=1),
            A.Normalize(mean=(0.45), std=(0.22), max_pixel_value=255),
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
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    train_dataloader = DataLoader(
        training_data, batch_size=1024, shuffle=True, num_workers=4
    )
    test_dataloader = DataLoader(
        test_data, batch_size=1024, shuffle=True, num_workers=4
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, True)
        test(test_dataloader, model, loss_fn)
        # if t % 5 == 0:
    print("Done!")

    torch.save(model.state_dict(), "model_weights.emnist.pth")
