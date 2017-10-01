import math

import torch as t
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import xavier_normal
from torch.nn.parameter import Parameter


class VariationalDropout(nn.Module):
    def __init__(self, input_size, out_size, log_alpha=-1.6094379124):
        """
        :param input_size: An int of input size
        :param out_size: An int of output size
        :param log_alpha: An float value of log of initial alpha value
               such that posterior over model parameters have form q(w_ij) = N(w_ij | theta_ij, alpha * theta_ij ^ 2)
               where thetha_ij is parameter of the layer
        """
        super(VariationalDropout, self).__init__()

        self.input_size = input_size
        self.out_size = out_size

        self.theta_weight = Parameter(t.FloatTensor(input_size, out_size))
        self.theta_bias = Parameter(t.Tensor(out_size))

        self.log_alpha = Parameter(t.FloatTensor(out_size).fill_(log_alpha))

        self.reset_parameters()

        self.c = [1.16145124, -1.50204118, 0.58629921]

        self.zeros = Variable(t.zeros(out_size))

    def reset_parameters(self):
        self.theta_weight = xavier_normal(self.theta_weight)

        stdv = 1. / math.sqrt(self.out_size)
        self.theta_bias.data.uniform_(-stdv, stdv)

    def kld(self, log_alpha, alpha):
        return -0.5 * log_alpha.sum() - t.stack([t.pow(alpha, power + 1) * self.c[power] for power in range(3)]).sum()

    def forward(self, input, train=False):
        """
        :param input: An float tensor with shape of [batch_size, input_size]
        :param train: An boolean value indicating whether forward propagation called when training is performed
        :return: An float tensor with shape of [batch_size, out_size] and negative layer-kld estimation
        """

        if train:

            eps = Variable(t.randn(self.input_size + 1, self.out_size))
            if input.is_cuda:
                eps.cuda()

            alpha = self.log_alpha.exp()
            noise = eps * alpha.sqrt()

            weight_noise = noise[:-1]
            bias_noise = noise[-1]

            weight = self.theta_weight * (1 + weight_noise)
            bias = self.theta_bias * (1 + bias_noise)

            return t.addmm(bias, input, weight), self.kld(self.log_alpha, alpha)

        return t.addmm(self.theta_bias, input, self.theta_weight)

    def clip_alpha(self):
        self.log_alpha = Parameter(t.min(self.log_alpha, self.zeros, 0).data)

