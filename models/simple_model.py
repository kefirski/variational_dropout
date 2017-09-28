import torch as t
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(784, 500),
            nn.ELU(),
            nn.Linear(500, 400),
            nn.ELU(),
            nn.Linear(400, 100),
            nn.ELU(),
            nn.Linear(100, 10)
        )

    def forward(self, input):
        """
        :param input: An float tensor with shape of [batch_size, 784]
        :return: An float tensor with shape of [batch_size, 10] filled with logits of likelihood
        """

        return self.fc(input)
