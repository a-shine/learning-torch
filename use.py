import random

import matplotlib.pyplot as plt
import torch.backends.mps

from architecture import NeuralNetwork
from data_load import load
from data_pre_process import TRANSFORMS

MODEL_PATH = "models/model_20231004-095818.pth"

device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Running model using {device} device")

# Load the test dataset
_, test_dataset = load()

# Take a random image from the test dataset
random_index = random.randint(0, len(test_dataset) - 1)
image, label = test_dataset[random_index]

# Pre-process the image
pre_process_image = TRANSFORMS(image).to(device)

# Create an instance of the model and move it to the device (GPU or CPU) and load the model parameters
model = NeuralNetwork((28 * 28), 10).to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()  # Set model to evaluation mode i.e. TODO: WHAT DOES THIS DO?

# Make a prediction
pred = model(pre_process_image)

# Plot the image, label and prediction
plt.imshow(image, cmap='gray')
plt.title(f"Label: {label}, Prediction: {pred.argmax(0)}")
plt.axis('off')
plt.show()
