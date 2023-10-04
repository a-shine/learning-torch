from torch import nn

class NeuralNetwork(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()

        # Sequential in a wrapper that allows us to create a sequential model by passing a list of layers
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

    # Define the order in which the layers are called i.e. the forward pass
    def forward(self, x):
        return self.net(x)
