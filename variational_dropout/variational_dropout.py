import math

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import xavier_normal
from torch.nn.parameter import Parameter


class VariationalDropout(nn.Module):
    def __init__(self, input_size, out_size, log_alpha=-0.6566749439):
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

        self.sigm_weight = Parameter(t.FloatTensor(input_size, out_size))
        self.sigm_bias = Parameter(t.Tensor(out_size))

        # one row in alpha for bias term
        self.log_alpha = Parameter(t.FloatTensor(input_size + 1, out_size).fill_(log_alpha))

        self.reset_parameters()

        self.k = [0.63576, 1.87320, 1.48695]

        self.zeros = Variable(t.zeros(out_size))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_size)

        self.theta_weight = xavier_normal(self.theta_weight)
        self.theta_bias.data.uniform_(-stdv, stdv)

        self.sigm_weight = xavier_normal(self.sigm_weight)
        self.sigm_bias.data.uniform_(-stdv, stdv)

    def kld(self, log_alpha, alpha):

        first_term = self.k[0] * F.sigmoid(self.k[1] + self.k[2] * log_alpha)
        second_term = 0.5 * t.log(1 + t.pow(alpha, -1))

        return (first_term - second_term - self.k[0]).sum().neg()

    def forward(self, input, train=False):
        """
        :param input: An float tensor with shape of [batch_size, input_size]
        :param train: An boolean value indicating whether forward propagation called when training is performed
        :return: An float tensor with shape of [batch_size, out_size] and negative layer-kld estimation
        """

        if not train:
            return t.addmm(self.theta_bias, input, self.theta_weight)

        eps = Variable(t.randn(self.input_size + 1, self.out_size))
        if input.is_cuda:
            eps.cuda()

        alpha = self.log_alpha.exp()
        noise = eps * alpha.sqrt()

        weight = self.theta_weight + self.sigm_weight * noise[:-1]
        bias = self.theta_bias + self.sigm_bias * noise[-1]

        return t.addmm(bias, input, weight), self.kld(self.log_alpha, alpha)
