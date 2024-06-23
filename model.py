import torch
from torch import nn
from motion import InfGenerator

class FNN(nn.Module):
    def __init__(self, type, input_dim=3, hidden_dim=10, output_dim=3):
        super().__init__()

        # Dimensions for input, hidden and output
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Learning rate definition
        self.learning_rate = 1e-4

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
        self.gen = InfGenerator(type=type)
        self.gen_type = type

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
    
    def normalize(self, x:torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=1).reshape(x.size(dim=0), 1)
    
    def normalize_(self, x:torch.Tensor) -> torch.Tensor:
        return x / x.norm()

    # Forward propagation
    def forward(self, X:torch.Tensor) -> torch.Tensor:
        m = nn.Tanh()
        self.q = X

        # First linear layer
        self.y1 = torch.matmul(X.T, self.w1)

        # First non-linearity
        self.y2 = m(self.y1)

        # Second linear layer
        self.y3 = torch.matmul(self.y2, self.w2)

        # Second non-linearity
        self.y4 = m(self.y3)

        # Third linear layer
        self.y5 = torch.matmul(self.y4, self.w3)

        # Third nonlinearity
        self.y6 = self.normalize(self.y5)
        # self.y6 = self.y5
        return self.y6.T
    
    # Vector field of Lie algebra
    def xi_Q(self, t:torch.Tensor) -> torch.Tensor:
        N = t.size(dim=2)
        return (self.gen.generator(t) @ self.y6.reshape(N,3,1)).reshape(N,3)
    
    def xi(self, t: torch.Tensor) -> torch.Tensor:
        N = t.size(dim=2)
        return (self.gen.generator_inv(t) @ self.y6.reshape(N,3,1)).reshape(N,3)
    
    # Distribution cost
    def J_dist(self, t: torch.Tensor) -> torch.Tensor:
        if self.gen_type == 'SE(2)':
            q_dot = self.sys.q_dot(t)[1:4]
        elif self.gen_type == 'S1xR2':
            q_dot = torch.index_select(self.sys.q_dot(t), dim=0, index=torch.tensor([0, 2, 3]))
        return (self.y6.T - torch.nn.functional.normalize(q_dot, dim=0)).norm()**2

    # Null space cost
    def J_null(self, t: torch.Tensor) -> torch.Tensor:
        return (self.y6 @ self.sys.q_dot(t)[1:4]).diag(0).norm()**2
    
    # Loss function
    def J_theta(self, t: torch.Tensor) -> torch.Tensor:

        # data size
        N = t.size(dim=2)

        # null space cost
        J1 = self.J_dist(t) / N
        
        # regularization
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

    def train(self, X, t):
        # Forward propagation
        xi = self.forward(X)

        # Backward propagation and gradient descent
        J = self.backward(t)

        return xi, J