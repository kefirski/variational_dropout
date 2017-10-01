import torch.nn as nn
import torch.nn.functional as F


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(784, 500),
            nn.ELU(),
            nn.Linear(500, 50),
            nn.ELU(),
            nn.Linear(50, 10)
        )

    def forward(self, input):
        """
        :param input: An float tensor with shape of [batch_size, 784]
        :return: An float tensor with shape of [batch_size, 10] filled with logits of likelihood
        """

        return self.fc(input)

    def loss(self, **kwargs):
        out = self(kwargs['input'])
        return F.cross_entropy(out, kwargs['target'], size_average=kwargs['average'])

