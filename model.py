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
        self.learning_rate_1 = 1e-5
        self.learning_rate_2 = 1e-5
        self.learning_rate_3 = 1e-5

        # Our parameters (weights)
        # w1: 3 x 100
        self.w1 = torch.ones(self.input_dim, self.hidden_dim, requires_grad=True)
        # self.w1 = self.w1*0.1

        # w2: 100 x 100
        self.w2 = torch.ones(self.hidden_dim, self.hidden_dim, requires_grad=True)
        # self.w2 = self.w2*0.1

        # w3: 100 x 3
        self.w3 = torch.ones(self.hidden_dim, self.output_dim, requires_grad=True)
        # self.w3 = self.w3*0.1

        # create infenitisimal generator
        self.gen = InfGenerator()

        # Initialize dynamical system
        self.sys = self.gen.sys

        # Momentum in gradient
        self.gamma = .3
        self.dJ_dw1_m = torch.zeros(self.w1.size())
        self.dJ_dw2_m = torch.zeros(self.w2.size())
        self.dJ_dw3_m = torch.zeros(self.w3.size())


    def sigmoid(self, s):
        return 1 / (1 + torch.exp(-s))

    def sigmoid_first_derivative(self, s):
        return s * (1 - s)

    # Forward propagation
    def forward(self, X:torch.Tensor):
        # First linear layer
        self.y1 = torch.matmul(X.T, self.w1)

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
        dy4_dy2 = torch.matmul(dy4_dy3, self.w2) # N x 100
        dy2_dy1 = self.sigmoid_first_derivative(self.y2) # N x 100
        dy4_dy1 = torch.matmul(dy4_dy2.T, dy2_dy1) # 100 x 100
        dy5_dy1 = torch.matmul(dy5_dy4.T, dy4_dy1) # 3 x 100
        dy5_dq = torch.matmul(dy5_dy1, self.w1.T) # 3 x 3
        ddt_dy5_dq = torch.matmul(dy5_dq, self.sys.q_dot(t)[1:4]) # 3 x N
        return ddt_dy5_dq
    
    # Loss function
    def J_theta(self, t: torch.Tensor) -> torch.Tensor:

        # data size
        N = t.size(dim=1)

        # nonholonomic momentum cost
        J1 =  (torch.matmul(self.sys.dL_dqdot(t)[1:4].T, self.d_dt_forward(t)).diag(0) - 
            torch.matmul(self.sys.d_dt_dL_dqdot(t)[1:4].T, self.gen.generator(t).matmul(self.y5.reshape(N,3,1)).reshape(N,3).T).diag(0) - 
            torch.matmul(self.sys.dL_dqdot(t)[1:4].T, self.gen.d_dt_generator(t).matmul(self.y5.reshape(N,3,1)).reshape(N,3).T).diag(0) -
            torch.matmul(self.sys.dL_dqdot(t)[1:4].T, self.gen.generator(t).matmul(self.d_dt_forward(t).T.reshape(N,3,1)).reshape(N,3).T).diag(0)).norm()
        
        # regularization
        # J2 = self.y5.norm()
        J2 = 0

        return J1 + J2

    # Backward propagation
    def backward(self, t) -> torch.Tensor:

        J1 = self.J_theta(t)
        J1.backward()

        dJ1_dw1 = self.w1.grad
        dJ1_dw2 = self.w2.grad
        dJ1_dw3 = self.w3.grad

        # breakpoint()
        # Gradient descent on the weights from our 3 linear layers
        with torch.no_grad():
            self.w1 -= self.learning_rate_1 * (dJ1_dw1 + self.gamma * self.dJ_dw1_m)
            self.w2 -= self.learning_rate_2 * (dJ1_dw2 + self.gamma * self.dJ_dw2_m)
            self.w3 -= self.learning_rate_3 * (dJ1_dw3 + self.gamma * self.dJ_dw3_m)

        self.dJ_dw1_m = dJ1_dw1
        self.dJ_dw2_m = dJ1_dw2
        self.dJ_dw3_m = dJ1_dw3
        
        return J1

    def train(self, X, t):
        # Forward propagation
        y5 = self.forward(X)

        # Backward propagation and gradient descent
        J = self.backward(t)

        return y5, J