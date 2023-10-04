import random

import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

from architecture import NeuralNetwork

device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device} device")

test_dataset = datasets.MNIST(root='./data', train=False, download=True)

# Test the model with a single image
# Take a random image from the test dataset
random_index = random.randint(0, len(test_dataset) - 1)
image, label = test_dataset[random_index]
plt.imshow(image.squeeze().numpy(), cmap='gray')
plt.title(f"Label: {label}")
plt.axis('off')
plt.show()

model = NeuralNetwork((28 * 28), 10).to(device)
model.load_state_dict(torch.load("model.pth"))

# Make a prediction
image = image.to(device)
pred = model(image)
print(f"Prediction: {pred.argmax(0)}")
