import random

import matplotlib.pyplot as plt
from torchvision import datasets


def load(verbose=False, visualise=False) -> (datasets.MNIST, datasets.MNIST):
    """
    Loads the MNIST dataset
    :return: training dataset and test dataset
    """
    training_dataset = datasets.MNIST(root='./data', train=True, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True)

    if verbose:
        # How much data?
        print(f"Length of training dataset: {len(training_dataset)}")
        print(f"Length of test dataset: {len(test_dataset)}")

        # What is it?
        print(f"First input entry: {training_dataset[0][0]}")
        print(f"First label entry: {training_dataset[0][1]}")

    # What does it look like?
    if visualise:
        plot_random(training_dataset)

    return training_dataset, test_dataset


def plot_random(dataset: datasets.MNIST):
    """
    Plot 5 random images from the dataset. Warning: this function will block the program until the plot is closed.
    :param dataset: the dataset from which to pick 5 random images
    """
    # What does the raw data look like?
    plt.figure(figsize=(1, 5))
    for i in range(5):
        random_index = random.randint(0, len(dataset) - 1)  # Take an image at random from the training dataset
        image, label = dataset[random_index]  # Training dataset is a tuple of (image, label)
        plt.subplot(1, 5, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(label)
        plt.axis('off')

    plt.show()
