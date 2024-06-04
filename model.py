import torch
from torch import nn
from motion import InfGenerator

class FNN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=10, output_dim=3):
        super().__init__()

        # Dimensions for input, hidden and output
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Learning rate definition
        self.learning_rate_1 = 1e-3
        self.learning_rate_2 = 1e-3
        self.learning_rate_3 = 1e-3

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


    def sigmoid(self, s):
        return 2 / (1 + torch.exp(-s)) - 1

    def sigmoid_first_derivative(self, s):
        return 1 * s * (1 - s)
    
    def normalize(self, x:torch.Tensor):
        return x / x.norm(dim=1).reshape(x.size(dim=0), 1)
    
    def normalize_(self, x:torch.Tensor):
        return x / x.norm()

    # Forward propagation
    def forward(self, X:torch.Tensor) -> torch.Tensor:
        m = nn.Sigmoid()
        self.q = X

        # First linear layer
        self.y1 = torch.matmul(X.T, self.w1)

        # First non-linearity
        # self.y2 = self.sigmoid(self.y1)
        self.y2 = m(self.y1)

        # Second linear layer
        self.y3 = torch.matmul(self.y2, self.w2)

        # Second non-linearity
        # self.y4 = self.sigmoid(self.y3)
        self.y4 = m(self.y3)

        # Third linear layer
        self.y5 = torch.matmul(self.y4, self.w3)

        # Third nonlinearity
        # self.y6 = self.normalize(self.sigmoid(self.y5)) # N x 3
        self.y6 = self.normalize(m(self.y5)) # N x 3
        # self.y6 = self.y5
        return self.y6.T
    
    # Forward propagation without updating
    def forward_(self, X:torch.Tensor):
        # First linear layer
        y1 = torch.matmul(X.T, self.w1)

        # First non-linearity
        y2 = self.sigmoid(y1)

        # Second linear layer
        y3 = torch.matmul(y2, self.w2)

        # Second non-linearity
        y4 = self.sigmoid(y3)

        # Third linear layer
        y5 = torch.matmul(y4, self.w3)

        # Third nonlinearity
        y6 = self.normalize_(self.sigmoid(y5)) # N x 3
        # y6 = y5
        return y6.T

    # Time derivative of forward function autograd
    def d_dt_forward_auto(self, t:torch.Tensor) -> torch.Tensor:
        # jac = torch.autograd.functional.jacobian(self.forward_, torch.ones(4), create_graph=True) # 3 x 4
        # df_dt = jac @ self.sys.q_dot(t) # 3 x N
        # return df_dt
        df_dt = torch.autograd.grad(self.y6.T, t, torch.ones_like(self.y6.T), retain_graph=True, create_graph=True)
        return df_dt[0].reshape(4, t.size(dim=2))
    
    # Momentum map
    def J_xi(self, t:torch.Tensor) -> torch.Tensor:
        N = t.size(dim=2)
        return torch.matmul(self.sys.dL_dqdot(t)[1:4].T, (self.gen.generator(t) @ self.y6.reshape(N,3,1)).reshape(N,3).T).diag(0) # 1 x N

    # Time derivative of momentum map
    def d_dt_J_xi(self, t:torch.Tensor) -> torch.Tensor:
        J_xi = self.J_xi(t)
        d_dt_J_xi = torch.autograd.grad(J_xi, t, torch.ones_like(J_xi), retain_graph=True, create_graph=True)
        # d_dt_J_xi = torch.autograd.functional.jacobian(self.J_xi, t, create_graph=True)
        return d_dt_J_xi[0].sum(0)
    
    # Loss function
    def J_theta(self, t: torch.Tensor) -> torch.Tensor:

        # data size
        N = t.size(dim=2)

        # d_dt_forward = self.d_dt_forward_auto(t)[1:4] # 3 x N

        # d_dt_J_xi = self.d_dt_J_xi(t)

        # nonholonomic momentum cost
        # J1 = 1 * (torch.matmul(self.sys.dL_dqdot(t)[1:4].T, d_dt_forward).diag(0) - d_dt_J_xi).norm()
        # J1 = d_dt_J_xi.norm()
        # J1 = 0
        
        # regularization
        # J2 = - 0.01 * self.y6[:,0].norm() - 1 * self.y6[:,1].norm() - 1 * self.y6[:,2].norm()
        # J2 = - 5 * d_dt_forward.norm()
        # J2 = 0

        # ground truth
        J = (torch.tensor([[.5], [0.], [0.]]).repeat(1,N) + self.sys.q(t)[3] * torch.tensor([[0.], [.5], [0.]]) - self.sys.q(t)[2] * torch.tensor([[0.], [0.], [.5]]) - self.y6.T).norm()

        # J = J1 + J2

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
            self.w1 -= self.learning_rate_1 * (dJ1_dw1 + self.gamma * self.dJ_dw1_m)
            self.w2 -= self.learning_rate_2 * (dJ1_dw2 + self.gamma * self.dJ_dw2_m)
            self.w3 -= self.learning_rate_3 * (dJ1_dw3 + self.gamma * self.dJ_dw3_m)

        self.dJ_dw1_m = dJ1_dw1
        self.dJ_dw2_m = dJ1_dw2
        self.dJ_dw3_m = dJ1_dw3

        return J1

    def train(self, X, t):
        # Forward propagation
        xi = self.forward(X)

        # Backward propagation and gradient descent
        J = self.backward(t)

        return xi, J