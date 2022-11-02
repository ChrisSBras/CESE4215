# pylint: disable=missing-function-docstring,missing-class-docstring,invalid-name
from torch import nn

class LenetMNIST(nn.Module):

    def __init__(self):
        super(LenetMNIST, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(6, 16, 5, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )


    def forward(self, x):
        return self.model(x)
