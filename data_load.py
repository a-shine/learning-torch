from torchvision import datasets
import matplotlib.pyplot as plt
import random


def load_data(visualise=False) -> (datasets.MNIST, datasets.MNIST):
    """
    Loads the MNIST dataset
    :return: training dataset and test dataset
    """
    training_dataset = datasets.MNIST(root='./data', train=True, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True)

    if visualise:
        explore_data(training_dataset, test_dataset)

    return training_dataset, test_dataset


def explore_data(training_dataset: datasets.MNIST, test_dataset: datasets.MNIST):
    """
    Explores the MNIST dataset to see what we're working with
    :param training_dataset: training dataset
    :param test_dataset: test dataset
    """
    # How much data?
    print(f"Length of training dataset: {len(training_dataset)}")
    print(f"Length of test dataset: {len(test_dataset)}")
    print(f"First input entry: {training_dataset[0][0]}")
    print(f"First label entry: {training_dataset[0][1]}")

    # What does the raw data look like?
    plt.figure(figsize=(1, 5))
    for i in range(5):
        random_index = random.randint(0, len(training_dataset) - 1)  # Take an image at random from the training dataset
        image, label = training_dataset[random_index]  # Training dataset is a tuple of (image, label)
        plt.subplot(1, 5, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(label)
        plt.axis('off')

    plt.show()
