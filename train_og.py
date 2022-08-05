import typing as t

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import DEVICE, synthesize_data


class StarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv0 = nn.Conv2d(1, 32, 3)
        self.conv1 = nn.Conv2d(32, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.conv4 = nn.Conv2d(256, 512, 3)
        self.conv5 = nn.Conv2d(512, 512, 3)

        self.fc1 = nn.Linear(512, 256)
        #change number of output to 6
        self.fc2 = nn.Linear(256, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv0(x)))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class StarDataset(torch.utils.data.Dataset):
    """Return star image and labels"""

    def __init__(self, data_size=50000):
        self.data_size = data_size

    def __len__(self) -> int:
        return self.data_size

    def __getitem__(self, idx) -> t.Tuple[torch.Tensor, torch.Tensor]:
        image, label = synthesize_data(has_star=True)
        return image[None], label


def train(model: StarModel, dl: StarDataset, num_epochs: int) -> StarModel:

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        print(f"EPOCH: {epoch}")
        losses = []
        for image, label in tqdm(dl, total=len(dl)):
            image = image.to(DEVICE).float()
            label = label.to(DEVICE).float()

            optimizer.zero_grad()

            preds = model(image)
            loss = loss_fn(preds, label)
            loss.backward()
            losses.append(loss.detach().cpu().numpy())
            optimizer.step()
        print(np.mean(losses))

    return model


def main():

    model = StarModel().to(DEVICE)

    star_model = train(
        model,
        torch.utils.data.DataLoader(StarDataset(), batch_size=64, num_workers=8),
        num_epochs=30,
    )
    torch.save(star_model.state_dict(), "model.pickle")


if __name__ == "__main__":
    main()
