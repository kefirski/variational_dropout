import torch.nn as nn
import torch.nn.functional as F


class DropoutModel(nn.Module):
    def __init__(self):
        super(DropoutModel, self).__init__()

        self.fc = nn.ModuleList([
            nn.Linear(784, 500),
            nn.Linear(500, 50),
            nn.Linear(50, 10)
        ])

    def forward(self, input, p=0):
        """
        :param input: An float tensor with shape of [batch_size, 784]
        :param p: An float value in [0, 1.] with probability of elements to be zeroed
        :return: An float tensor with shape of [batch_size, 10] filled with logits of likelihood
        """

        result = input

        for i, layer in enumerate(self.fc):
            result = F.elu(layer(result))

            if i < len(self.fc) - 1:
                result = F.dropout(result, p, training=True)

        return result

    def loss(self, **kwargs):
        out = self(kwargs['input'], kwargs['p'])
        return F.cross_entropy(out, kwargs['target'], size_average=kwargs['average'])
