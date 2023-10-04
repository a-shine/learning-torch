import random

import matplotlib.pyplot as plt
import torch
from architecture import NeuralNetwork
from data_load import load_data
from pre_process import TRANSFORMS

MODEL_PATH = "models/model_20231004-093408.pth"

device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device} device")

# Load the test dataset
_, test_dataset = load_data()

# Test the model with a single image
# Take a random image from the test dataset
random_index = random.randint(0, len(test_dataset) - 1)
image, label = test_dataset[random_index]
plt.imshow(image, cmap='gray')
plt.title(f"Label: {label}")
plt.axis('off')
plt.show()

pre_process_image = TRANSFORMS(image).to(device)

model = NeuralNetwork((28 * 28), 10).to(device)
model.load_state_dict(torch.load(MODEL_PATH))

pred = model(pre_process_image)
print(f"Prediction: {pred.argmax(0)}")
