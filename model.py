import torch
from torch import nn
from motion import EquationsOfMotion

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

        # w2: 100 x 3
        self.w2 = torch.randn(self.hidden_dim, self.output_dim, requires_grad=True)

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
        y4 = self.sigmoid(self.y3)
        return y4
    
    # Time derivative of forward function
    def d_dt_forward(self, y4, t):
        dy4_dy3 = self.sigmoid_first_derivative(y4) # N x 3
        dy4_dy2 = torch.matmul(dy4_dy3, torch.t(self.w2)) # N x 100
        dy2_dy1 = self.sigmoid_first_derivative(self.y2) # N x 100
        dy4_dy1 = torch.matmul(torch.t(dy4_dy2), dy2_dy1) # 100 x 100
        return torch.matmul(dy4_dy1, self.w1) # 100 x 3
    
    # Loss function
    def J_theta(self):
        return

    # Backward propagation
    def backward(self, X, l, y4):
        # Derivative of binary cross entropy cost w.r.t. final output y4
        self.dC_dy4 = y4 - l

        '''
        Gradients for w2: partial derivative of cost w.r.t. w2
        dC/dw2
        '''
        self.dy4_dy3 = self.sigmoid_first_derivative(y4)
        self.dy3_dw2 = self.y2

        # Y4 delta: dC_dy4 dy4_dy3
        self.y4_delta = self.dC_dy4 * self.dy4_dy3

        # This is our gradients for w1: dC_dy4 dy4_dy3 dy3_dw2
        self.dC_dw2 = torch.matmul(torch.t(self.dy3_dw2), self.y4_delta)

        '''
        Gradients for w1: partial derivative of cost w.r.t w1
        dC/dw1
        '''
        self.dy3_dy2 = self.w2
        self.dy2_dy1 = self.sigmoid_first_order_derivative(self.y2)

        # Y2 delta: (dC_dy4 dy4_dy3) dy3_dy2 dy2_dy1
        self.y2_delta = torch.matmul(self.y4_delta, torch.t(self.dy3_dy2)) * self.dy2_dy1

        # Gradients for w1: (dC_dy4 dy4_dy3) dy3_dy2 dy2_dy1 dy1_dw1
        self.dC_dw1 = torch.matmul(torch.t(X), self.y2_delta)

        # Gradient descent on the weights from our 2 linear layers
        self.w1 -= self.learning_rate * self.dC_dw1
        self.w2 -= self.learning_rate * self.dC_dw2

    def train(self, X, l):
        # Forward propagation
        y4 = self.forward(X)

        # Backward propagation and gradient descent
        self.backward(X, l, y4)