import math

import torch as t
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import xavier_normal
from torch.nn.parameter import Parameter


class VariationalDropout(nn.Module):
    def __init__(self, input_size, out_size, log_alpha=-0.6931471806):
        """
        :param input_size: An int of input size
        :param out_size: An int of output size
        :param log_alpha: An float value of log of initial alpha value
               such that posterior over model parameters have form q(w_ij) = N(w_ij | theta_ij, alpha * theta_ij ^ 2)
               thetha_ij is parameter of the layer
        """
        super(VariationalDropout, self).__init__()

        self.input_size = input_size
        self.out_size = out_size

        self.theta = Parameter(t.FloatTensor(input_size, out_size))
        self.bias = Parameter(t.Tensor(out_size))

        self.log_alpha = Parameter(t.FloatTensor(out_size).fill_(log_alpha))

        self.reset_parameters()

        self.c = [1.16145124, -1.50204118, 0.58629921]

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_size)

        self.theta = xavier_normal(self.theta)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def kld(self, log_alpha, alpha):
        return -0.5 * log_alpha.sum() - t.stack([t.pow(alpha, power) * self.c[power] for power in range(3)]).sum()

    def forward(self, input, train=False):
        """
        :param input: An float tensor with shape of [batch_size, input_size]
        :param train: An boolean value indicating whether forward propagation called when training is performed
        :return: An float tensor with shape of [batch_size, out_size] and negative layer-kld estimation
        """

        '''
        Since dropout is approximated with Local Reparameterization Trick,
        we firsly have to obtain mu and std of resulting Gaussian, 
        and then perform sampling of the result from it
        '''

        mu = t.addmm(self.bias, input, self.theta)
        if not train:
            '''
            Expectation over models from posterior is approximated with mean value.
            Better results could be achived by MC approximation.
            '''
            return mu

        alpha = self.log_alpha.exp()
        std = t.addmm(self.bias.abs, input.abs, self.theta.abs()) * alpha.sqrt()

        eps = Variable(t.randn(*mu.size()))
        if mu.is_cuda:
            eps = eps.cuda()

        return eps * std + mu, self.kld(self.log_alpha, alpha)
