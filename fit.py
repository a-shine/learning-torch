import os
import time

import torch
import torch.nn as nn
import torch.backends.mps
from torch.utils.data import DataLoader

from architecture import NeuralNetwork
from data_load import load
from data_pre_process import pre_process_datasets

# Parameters
# https://ai.stackexchange.com/questions/8560/how-do-i-choose-the-optimal-batch-size
# https://stats.stackexchange.com/questions/164876/what-is-the-trade-off-between-batch-size-and-number-of-iterations-to-train-a-neu
# It has been observed that with larger batch there is a significant degradation in the quality of the model, as
# measured by its ability to generalize i.e. large batch size is better for training but not for generalization
# (overfitting)
BATCH_SIZE = 2 ** 5
EPOCHS = 15

# Detect device for training and running the model
# Installing CUDA - https://docs.nvidia.com/cuda/cuda-quick-start-guide/
device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training/fitting using {device} device")

# Load the MNIST dataset
training_dataset, testing_dataset = load(verbose=True, visualise=False)

# Pre-process the data (i.e. flatten the images into a single vector of pixels)
processed_training_data, processed_test_data = pre_process_datasets(training_dataset, testing_dataset, verbose=True)


# Create a data loader to handle loading data in and out of memory in batches

# Create data loaders.
train_dataloader = DataLoader(processed_training_data, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(processed_test_data, batch_size=BATCH_SIZE)

# Look at the shape of the data coming out of the data loader (batch size, channels, height, width)
for X, y in test_dataloader:
    print(f"Shape of X [N, W*H]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Create an instance of the model and move it to the device (GPU or CPU)
model = NeuralNetwork((28 * 28), 10).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # start with a high learning rate and allow it to decay
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)


def fit(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()  # Set model to training mode TODO (what does this do?)
    # Go through each batch in the data loader (i.e. all training data),
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)  # compute the loss per batch

        # Backpropagation
        loss.backward()  # compute the gradient of the loss with respect to the model parameters
        optimizer.step()  # adjust the parameters by the computed gradients
        optimizer.zero_grad()  # TODO (what does this do?)

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
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
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


for t in range(EPOCHS):
    print(f"Epoch {t + 1}\n-------------------------------")
    fit(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

# Save model in models folder, create if it doesn't exist
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), f"models/model_{time.strftime('%Y%m%d-%H%M%S')}.pth")
print("Saved PyTorch model state in models folder")
