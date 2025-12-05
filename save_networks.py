import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
import kagglehub
import os

# %%
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
pmem = True if device == "cuda" else False
print(device)

# %%
transform = transforms.Compose([transforms.ToTensor()])
images = torchvision.datasets.ImageFolder("coil-20-proc", transform=transform)
print(images.classes)
train_images = DataLoader(images, batch_size=72, shuffle=True, pin_memory=pmem)

# %%
def label_string(label_int):
    match label_int:
        case 0:
            return "duck"
        case 1:
            return "block toy"
        case 2:
            return "toy racecar"
        case 3:
            return "waving cat"
        case 4:
            return "anacin box"
        case 5:
            return "toy convertible"
        case 6:
            return "block toy 2"
        case 7:
            return "baby powder"
        case 8:
            return "tylenol"
        case 9:
            return "vaseline"
        case 10:
            return "block toy (semicircle)"
        case 11:
            return "cup"
        case 12:
            return "piggy bank"
        case 13:
            return "valve"
        case 14:
            return "bucket"
        case 15:
            return "conditioner bottle"
        case 16:
            return "pot"
        case 17:
            return "teacup"
        case 18:
            return "toy convertible (lame)"
        case 19:
            return "cream cheese tub"
        case _:
            return "none of the above"

# %%

# %%
def save_model(p):
    model = nn.Sequential()
    model.add_module(
        'conv1',
        nn.Conv2d(
            in_channels=3,out_channels=32,
            kernel_size=5,padding=2
        )
    )
    model.add_module('relu1',nn.ReLU())
    model.add_module('pool1',nn.MaxPool2d(kernel_size=2))

    model.add_module(
        'conv2',
        nn.Conv2d(
            in_channels=32, out_channels=64,
            kernel_size=5,padding=2
        )
    )
    model.add_module('relu2',nn.ReLU())
    model.add_module('pool2',nn.MaxPool2d(kernel_size=2))

    torch.save(model.state_dict(), "test.pth")

    model.add_module('flatten',nn.Flatten())

    x = torch.ones((4,3,128,128))
    dims = model(x).shape

    model.add_module('fc1',nn.Linear(dims[1], 1024))
    model.add_module('relu3', nn.ReLU())
    model.add_module('dropout',nn.Dropout(p=0.5))
    model.add_module('fc2',nn.Linear(1024,20))
    model.to(device)

    # %%
    num_epochs = 10

    loss_history_train = np.zeros(num_epochs)
    accuracy_history_train = np.zeros(num_epochs)
    loss_history_valid = np.zeros(num_epochs)
    accuracy_history_valid = np.zeros(num_epochs)

    # %%
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # %%
    torch.manual_seed(540)

    noise_std = p

    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_images:
            # test adding noise:
            noise = torch.randn_like(x_batch)*noise_std
            x_batch = x_batch + noise

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            loss_history_train[epoch] += loss.item()*y_batch.size(0)
            accuracy_history_valid[epoch] += is_correct.sum()

        model.eval()
        print(epoch)

    torch.save(model.state_dict(), "prebuilts/coil20_{:.3f}.pth".format(noise_std))
    torch.mps.empty_cache()


for i in range(0,101):
    save_model(i/1000)