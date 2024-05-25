import torch
from torch import nn
from motion import InfGenerator

class FNN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=100, output_dim=3):
        super().__init__()

        # Dimensions for input, hidden and output
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Learning rate definition
        self.learning_rate = 0.001

        # Our parameters (weights)
        # w1: 3 x 100
        self.w1 = torch.randn(self.input_dim, self.hidden_dim, requires_grad=True)

        # w2: 100 x 100
        self.w2 = torch.randn(self.hidden_dim, self.hidden_dim, requires_grad=True)

        # w3: 100 x 3
        self.w3 = torch.randn(self.hidden_dim, self.output_dim, requires_grad=True)

        # create infenitisimal generator
        self.gen = InfGenerator()

        # Initialize dynamical system
        self.sys = self.gen.sys


    def sigmoid(self, s):
        return 1 / (1 + torch.exp(-s))

    def sigmoid_first_derivative(self, s):
        return s * (1 - s)

    # Forward propagation
    def forward(self, X):
        # First linear layer
        self.y1 = torch.matmul(X, self.w1)

        # First non-linearity
        self.y2 = self.sigmoid(self.y1)

        # Second linear layer
        self.y3 = torch.matmul(self.y2, self.w2)

        # Second non-linearity
        self.y4 = self.sigmoid(self.y3)

        # Third linear layer
        self.y5 = torch.matmul(self.y4, self.w3)
        return self.y5
    
    # Time derivative of forward function
    def d_dt_forward(self, t):
        dy5_dy4 = self.w3 # 100 x 3
        dy4_dy3 = self.sigmoid_first_derivative(self.y4) # N x 100
        dy4_dy2 = torch.matmul(dy4_dy3, self.w2.T) # 100 x 100
        dy2_dy1 = self.sigmoid_first_derivative(self.y2) # N x 100
        dy4_dy1 = torch.matmul(dy4_dy2.T, dy2_dy1) # 100 x 100
        dy5_dy1 = torch.matmul(dy5_dy4.T, dy4_dy1) # 3 x 100
        dy5_dq = torch.matmul(dy5_dy1, self.w1.T) # 3 x 3
        ddt_dy5_dq = torch.matmul(dy5_dq, self.sys.q_dot(t)[1:4]) # 3 x N
        return ddt_dy5_dq
    
    # Loss function
    def J_theta(self, t):
        return 1/2 * torch.pow(torch.matmul(self.sys.dL_dqdot(t)[1:4].T, self.d_dt_forward(t)) - 
                               torch.matmul(self.sys.d_dt_dL_dqdot(t), self.gen.generator(t).matmul(self.y5.T)) - 
                               torch.matmul(self.sys.dL_dqdot(t), self.gen.d_dt_generator(t).matmul(self.y5.T)) -
                               torch.matmul(self.sys.dL_dqdot(t), self.gen.generator(t).matmul(self.d_dt_forward(t))), 2)

    # Backward propagation
    def backward(self, t):
        J = self.J_theta(t)
        J.backward()

        dJ_dw1 = self.w1.grad
        dJ_dw2 = self.w2.grad
        dJ_dw3 = self.w3.grad

        # Gradient descent on the weights from our 3 linear layers
        self.w1 -= self.learning_rate * dJ_dw1
        self.w2 -= self.learning_rate * dJ_dw2
        self.w3 -= self.learning_rate * dJ_dw3

    def train(self, X, t):
        # Forward propagation
        y5 = self.forward(X)

        # Backward propagation and gradient descent
        self.backward(t)