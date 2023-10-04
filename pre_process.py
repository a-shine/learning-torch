import torch
from torchvision.transforms import ToTensor, Compose

from data_load import load_data


def pre_process(training_dataset, test_dataset, verbose=False) -> (torch.utils.data.Dataset, torch.utils.data.Dataset):
    """
    Pre-processes the MNIST dataset
    :return: training dataset and test dataset
    """

    # Data processing pipeline
    transforms = Compose([
        ToTensor(),
        torch.flatten
    ])

    # for every image in the training dataset and test dataset, apply the transforms
    processed_training_data = []
    processed_test_data = []

    for i in range(len(training_dataset)):
        image, label = training_dataset[i]
        image = transforms(image)
        processed_training_data.append((image, label))

    for i in range(len(test_dataset)):
        image, label = test_dataset[i]
        image = transforms(image)
        processed_test_data.append((image, label))

    if verbose:
        print(f"Length of training dataset: {len(processed_training_data)}")
        print(f"Length of test dataset: {len(processed_test_data)}")
        print(f"Shape of processed single input: {processed_training_data[0][0].shape}")


    return processed_training_data, processed_test_data
