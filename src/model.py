import torch
import torch.nn as nn
import torch.nn.functional as F


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.BatchNorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.BatchNorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.BatchNorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.BatchNorm4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3)
        self.BatchNorm5 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(256 * 5 * 5, 512)
        self.BatchNorm6 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, num_classes)
        #self.BatchNorm7 = nn.BatchNorm1d(1024)
        #self.fc3 = nn.Linear(1024, num_classes)
        
        # drop-out
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        x = F.relu(self.conv1(x))
        x = self.BatchNorm1(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.BatchNorm2(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.BatchNorm3(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = F.relu(self.conv4(x))
        x = self.BatchNorm4(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = F.relu(self.conv5(x))
        x = self.BatchNorm5(x)
        x = self.pool(x)
        
        x = x.view(-1, 256 * 5 * 5)
        
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.BatchNorm6(x)
        x = self.dropout(x)
        x = self.fc2(x)
        #x = self.BatchNorm7(x)
        #x = self.dropout(x)
        #x = self.fc3(x)
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
