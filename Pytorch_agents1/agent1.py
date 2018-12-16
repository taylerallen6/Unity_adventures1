import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import time



dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # run on GPU


class Network(torch.nn.Module):
	def __init__(self, D_in, H, D_out):
		super(Network, self).__init__()
		# random weights and biases.
		# set requires_grad=True so we CAN compute gradients with respect
		# to these tensors during backprop.
		self.w1 = torch.randn(H, D_in, device=device, dtype=dtype, requires_grad=True)
		self.b1 = torch.randn(H, device=device, dtype=dtype, requires_grad=True)

		self.w2 = torch.randn(H, H, device=device, dtype=dtype, requires_grad=True)
		self.b2 = torch.randn(H, device=device, dtype=dtype, requires_grad=True)

		self.w3 = torch.randn(H, H, device=device, dtype=dtype, requires_grad=True)
		self.b3 = torch.randn(H, device=device, dtype=dtype, requires_grad=True)
		
		self.w4 = torch.randn(D_out, H, device=device, dtype=dtype, requires_grad=True)
		self.b4 = torch.randn(D_out, device=device, dtype=dtype, requires_grad=True)

	def forward(self, x):

		# start_time = time.time()

		# forward pass
		y_pred = F.linear(x, self.w1, self.b1)
		y_pred = F.relu(y_pred)
		# print(y_pred.shape)

		y_pred = F.linear(y_pred, self.w2, self.b2)
		y_pred = F.relu(y_pred)
		# print(y_pred.shape)

		y_pred = F.linear(y_pred, self.w3, self.b3)
		y_pred = F.relu(y_pred)
		# print(y_pred.shape)

		y_pred = F.linear(y_pred, self.w4, self.b4)
		# print(y_pred.shape)
		# print()


		# elapsed_time = time.time() - start_time
		# print("epoch time: ", elapsed_time)

		return y_pred



def train(model, x, y, n_train_loops, n_epochs=2):
	learning_rate = .00001
	for e in range(n_epochs):
		print("epoch: ",e)

		# start_time = time.time()
		
		for t in range(n_train_loops):
			inputs, labels = x, y
			inputs, labels = Variable(inputs), Variable(labels)

			# reformat data
			# inputs = inputs.squeeze().reshape(inputs.shape[0], inputs.shape[2]*inputs.shape[3])

			# forward pass
			y_pred = model(inputs)

			print("y: ", y)
			print("y_pred: ", y_pred)

			predicted = torch.argmax(y_pred.data, dim=1)

			# compute loss
			loss = (y_pred - y).pow(2).sum()
			print(t, loss.item())
			print()

			# use autograd to compute backprop.
			# computes gradient of loss with respect to all tensors with requires_grad=True.
			# w1.grad, w2.grad, b1.grad, and b2.grad will be tensors of gradient of loss with respect to w1 and w2.
			loss.backward()
			
			# wrap in torch.no_grad() so we don't track weights in
			# autograd while updating them.
			with torch.no_grad():
				model.w1 -= learning_rate * model.w1.grad
				model.b1 -= learning_rate * model.b1.grad
				model.w2 -= learning_rate * model.w2.grad
				model.b2 -= learning_rate * model.b2.grad
				model.w3 -= learning_rate * model.w3.grad
				model.b3 -= learning_rate * model.b3.grad
				model.w4 -= learning_rate * model.w4.grad
				model.b4 -= learning_rate * model.b4.grad

				# zero out gradients for the next pass after updating them.
				model.w1.grad.zero_()
				model.b1.grad.zero_()
				model.w2.grad.zero_()
				model.b2.grad.zero_()
				model.w3.grad.zero_()
				model.b3.grad.zero_()
				model.w4.grad.zero_()
				model.b4.grad.zero_()

		# elapsed_time = time.time() - start_time
		# print("epoch time: ", elapsed_time)



# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.

BATCH_SIZE = 1
N, D_in, H, D_out = BATCH_SIZE, 20, 50, 10
n_epochs = 1

# random tensor for input and output data.
# leaving requires_grad=False because we don't need to compute gradients here.
x = torch.rand(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)
# print(x)
# print(y)


model = Network(D_in, H, D_out)

train(model, x, y, 100, n_epochs)