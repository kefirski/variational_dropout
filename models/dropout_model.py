import torch.nn as nn
import torch.nn.functional as F


class DropoutModel(nn.Module):
    def __init__(self):
        super(DropoutModel, self).__init__()

        self.fc = nn.ModuleList([
            nn.Linear(784, 500),
            nn.Linear(500, 400),
            nn.Linear(400, 100),
            nn.Linear(100, 10)
        ])

    def forward(self, input, p=0):
        """
        :param input: An float tensor with shape of [batch_size, 784]
        :param p: An float value in [0, 1.] with probability of elements to be zeroed
        :return: An float tensor with shape of [batch_size, 10] filled with logits of likelihood
        """

        result = input

        for i, layer in enumerate(self.fc):
            result = F.elu(layer(input))

            if i < len(self.fc) - 1:
                result = F.dropout(result, p, training=True)

        return result
