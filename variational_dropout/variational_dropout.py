import math

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import xavier_normal
from torch.nn.parameter import Parameter


class VariationalDropout(nn.Module):
    def __init__(self, input_size, out_size, log_sigma2=-4.6, threshold=3):
        """
        :param input_size: An int of input size
        :param log_sigma2: Initial value of log sigma ^ 2.
               It is crusial for training since it determines initial value of alpha
        :param threshold: Value for thresholding of validation. If log_alpha > threshold, then weight is zeroed
        :param out_size: An int of output size
        """
        super(VariationalDropout, self).__init__()

        self.input_size = input_size
        self.out_size = out_size

        self.theta = Parameter(t.FloatTensor(input_size, out_size))
        self.bias = Parameter(t.Tensor(out_size))

        self.log_sigma2 = Parameter(t.FloatTensor(input_size, out_size).fill_(log_sigma2))

        self.reset_parameters()

        self.k = [0.63576, 1.87320, 1.48695]

        self.threshold = threshold

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_size)

        self.theta.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    @staticmethod
    def clip(input, to=7):
        input = input.masked_fill(input < -to, -to)
        input = input.masked_fill(input > to, to)

        return input

    def kld(self, log_alpha):

        first_term = self.k[0] * F.sigmoid(self.k[1] + self.k[2] * log_alpha)
        second_term = 0.5 * t.log(1 + t.exp(-log_alpha))

        return -(first_term - second_term - self.k[0]).sum()

    def forward(self, input, train=False):
        """
        :param input: An float tensor with shape of [batch_size, input_size]
        :param train: An boolean value indicating whether forward propagation called of training is performed
        :return: An float tensor with shape of [batch_size, out_size] and negative layer-kld estimation
        """

        log_alpha = self.clip(self.log_sigma2 - self.theta ** 2)
        kld = self.kld(log_alpha)

        if not train:
            mask = log_alpha > self.threshold
            return t.addmm(self.bias, input, self.theta.masked_fill(mask, 0))

        mu = t.mm(input, self.theta)
        std = t.sqrt(t.mm(input ** 2, self.log_sigma2.exp()) + 1e-6)

        eps = Variable(t.randn(*mu.size()))
        if input.is_cuda:
            eps = eps.cuda()

        return std * eps + mu + self.bias, kld
