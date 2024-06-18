import torch
from torch import nn
import numpy as np
from motion import InfGenerator
from torchdiffeq import odeint

class FNN(nn.Module):
    def __init__(self, t: torch.Tensor, input_dim=3, hidden_dim=50, output_dim=3):
        super().__init__()

        # Time series
        self.t = t

        # Dimensions for input, hidden and output
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Learning rate definition
        self.learning_rate = 1e-3

        # Our parameters (weights)
        # w1: 3 x 100
        self.w1 = torch.FloatTensor(self.input_dim, self.hidden_dim).uniform_(-1, 1) / self.hidden_dim
        self.w1.requires_grad_()

        # w2: 100 x 100
        self.w2 = torch.FloatTensor(self.hidden_dim, self.hidden_dim).uniform_(-1, 1) / self.hidden_dim
        self.w2.requires_grad_()

        # w3: 100 x 3
        self.w3 = torch.FloatTensor(self.hidden_dim, self.output_dim).uniform_(-1, 1) / self.output_dim
        self.w3.requires_grad_()

        # create infenitisimal generator
        self.gen = InfGenerator()

        # Initialize dynamical system
        self.sys = self.gen.sys

        # Momentum in gradient
        self.gamma = 0.1
        self.dJ_dw1_m = torch.zeros(self.w1.size())
        self.dJ_dw2_m = torch.zeros(self.w2.size())
        self.dJ_dw3_m = torch.zeros(self.w3.size())

    
    def normalize(self, x:torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=1).reshape(x.size(dim=0), 1)
    
    def normalize_(self, x:torch.Tensor) -> torch.Tensor:
        return x / x.norm()

    # Forward propagation: time derivative of Lie algebra
    def forward(self, xi_0, t: torch.Tensor) -> torch.Tensor:
        m = nn.Tanh()

        TQ = torch.cat([self.sys.q(t), self.sys.q_dot(t)], dim=0) 

        # First linear layer
        self.y1 = torch.matmul(TQ.T, self.w1)

        # First non-linearity
        self.y2 = m(self.y1)

        # Second linear layer
        self.y3 = torch.matmul(self.y2, self.w2)

        # Second non-linearity
        self.y4 = m(self.y3)

        # Third linear layer
        self.y5 = torch.matmul(self.y4, self.w3)

        # Third nonlinearity
        # self.y6 = self.normalize(self.sigmoid(self.y5)) # N x 3
        # self.y6 = self.normalize(m(self.y5)) # N x 3
        # self.y6 = self.normalize(self.y5)
        # self.y6 = m(self.y5)
        self.y6 = self.y5
        return self.y6.T
    
    # Lie algebra element xi
    def xi(self, t: torch.Tensor) -> torch.Tensor:
        xi_0 = self.y6[0]
        breakpoint()
        self._xi = odeint(self.forward, xi_0, t, method='dopri5') # numerical integration of NN
        return self._xi
    
    # Vector field of Lie algebra
    def xi_Q(self, t:torch.Tensor) -> torch.Tensor:
        N = t.size(dim=0)
        return (self.gen.generator(t) @ self.xi(t).T.reshape(N,3,1)).reshape(N,3)
    
    def d_dt_xi_Q(self, t: torch.Tensor) -> torch.Tensor:
        N = t.size(dim=0)
        return (self.gen.generator(t) @ self.y6.T.reshape(N,3,1)).reshape(N,3).T + (self.gen.d_dt_generator(t) @ self.xi(t).reshape(N,3,1)).reshape(N,3).T
    
    # Momentum map
    def J_xi(self, t:torch.Tensor) -> torch.Tensor:
        return (self.sys.dL_dqdot(t)[1:4] * self.xi_Q(t).T).sum(0) # 1 x N
    
    # Time derivative of momentum map using the chain rule
    def d_dt_J_xi_chain(self, t:torch.Tensor) -> torch.Tensor:
        return self.sys.d_dt_dL_dqdot(t)[0][1:4] * self.xi_Q(t).T + self.sys.dL_dqdot(t)[1:4] * self.d_dt_xi_Q(t)
    
    # Nonholonomic momentum cost
    def J_nhc(self, t: torch.Tensor) -> torch.Tensor:
        d_dt_xi = self.d_dt_xi(t) # 3 x N
        d_dt_J_xi = self.d_dt_J_xi(t)
        return 1 * ((self.sys.dL_dqdot(t)[1:4] * d_dt_xi).sum(0) - d_dt_J_xi.sum(0)).norm()
    
    # Loss function
    def J_theta(self, t: torch.Tensor) -> torch.Tensor:

        # data size
        N = t.size(dim=0)

        # nonholonomic momentum cost
        # J1 = self.J_nhc(t)
        # J1 = (self.sys.dL_dqdot(t)[1:4] * self.d_dt_xi(t)).sum(0).norm()
        # J1 = self.d_dt_J_xi(t).sum(0).norm()
        # J1 = 0

        # null space cost
        J1 = self.J_nullspace(t)
        
        # regularization
        # J2 =  - 0.6 * self.y6.norm()
        # J2 = torch.matmul(self.sys.dL_dqdot(t)[1:4].T, d_dt_forward).diag(0).norm()
        # J2 = - 0.1 * self.xi_Q(t).norm()
        # J2 = - 1 * self.d_dt_xi(t).norm()
        J2 = 0

        # ground truth
        # J2 = (torch.tensor([[1.], [0.], [0.]]).repeat(1,N) + self.sys.y(t[2]) * torch.tensor([[0.], [1.], [0.]]) - self.sys.x(t[3]) * torch.tensor([[0.], [0.], [1.]]) - self.y6.T).norm()

        J = J1 + J2

        return J

    # Backward propagation
    def backward(self, t:torch.Tensor) -> torch.Tensor:
        
        J1 = self.J_theta(t)
        dJ_dw = torch.autograd.grad(J1, (self.w1, self.w2, self.w3))

        dJ1_dw1 = dJ_dw[0]
        dJ1_dw2 = dJ_dw[1]
        dJ1_dw3 = dJ_dw[2]

        # breakpoint()
        # Gradient descent on the weights from our 3 linear layers
        with torch.no_grad():
            self.w1 -= self.learning_rate * (dJ1_dw1 + self.gamma * self.dJ_dw1_m)
            self.w2 -= self.learning_rate * (dJ1_dw2 + self.gamma * self.dJ_dw2_m)
            self.w3 -= self.learning_rate * (dJ1_dw3 + self.gamma * self.dJ_dw3_m)

        self.dJ_dw1_m = dJ1_dw1
        self.dJ_dw2_m = dJ1_dw2
        self.dJ_dw3_m = dJ1_dw3

        return J1

    def train(self, t):
        
        xi_0 = torch.tensor(0)

        # Forward propagation
        xi = self.forward(xi_0, t)

        # Backward propagation and gradient descent
        J = self.backward(t)

        return xi, J