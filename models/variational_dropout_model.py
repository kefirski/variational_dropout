import torch as t
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
	
	def forward(self, input):
		"""
        :param input: An float tensor with shape of [batch_size, 784]
		:return: An float tensor with shape of [batch_size, 10] 
		         filled with logits of likelihood and kld estimation
        """

		result = input
		kld = 0

		for i, layer in enumerate(self.fc):
			if i < len(self.fc) - 1:
				result, kld = layer(input)
				result = F.elu(result)
				kld += kld
			else:
				return layer(result), kld
