import torch.nn as nn
import torch.nn.functional as F

from variational_dropout.variational_dropout import VariationalDropout


class VariationalDropoutModel(nn.Module):
    def __init__(self):
        super(VariationalDropoutModel, self).__init__()

        self.fc = nn.ModuleList([
            VariationalDropout(784, 500),
            VariationalDropout(500, 400),
            VariationalDropout(400, 100),
            nn.Linear(100, 10)
        ])

    def forward(self, input, train=False):
        """
        :param input: An float tensor with shape of [batch_size, 784]
        :param train: An boolean value indicating whether forward propagation called when training is performed
        :return: An float tensor with shape of [batch_size, 10]
                 filled with logits of likelihood and kld estimation
        """

        if train:

            result = input
            kld = 0

            for i, layer in enumerate(self.fc):
                if i < len(self.fc) - 1:
                    result, kld = layer(result)
                    result = F.elu(result)
                    kld += kld
                else:
                    return layer(result), kld
        else:

            result = input

            for i, layer in enumerate(self.fc):
                result = layer(result)
                if i < len(self.fc) - 1:
                    result = F.elu(result)

            return result
