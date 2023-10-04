import torch.utils.data
from torchvision.transforms import ToTensor, Compose

# Data processing pipeline
TRANSFORMS = Compose([
    ToTensor(),
    torch.flatten
])


def pre_process_datasets(training_dataset, test_dataset, verbose=False) -> (
        torch.utils.data.Dataset, torch.utils.data.Dataset):
    """
    Pre-processes the MNIST dataset
    :return: training dataset and test dataset
    """

    # for every image in the training dataset and test dataset, apply the transforms
    processed_training_data = apply_transformations(training_dataset)
    processed_test_data = apply_transformations(test_dataset)

    if verbose:
        print(f"Length of training dataset: {len(processed_training_data)}")
        print(f"Length of test dataset: {len(processed_test_data)}")
        print(f"Shape of processed single input: {processed_training_data[0][0].shape}")

    return processed_training_data, processed_test_data


def apply_transformations(dataset) -> torch.utils.data.Dataset:
    """
    Applies the transformations to a given dataset
    :param dataset: the dataset to apply the transformations to
    :return: the transformed dataset
    """

    # for every image in the training dataset and test dataset, apply the transforms
    processed_data = []

    for i in range(len(dataset)):
        image, label = dataset[i]
        image = TRANSFORMS(image)
        processed_data.append((image, label))

    return processed_data
