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
        self.learning_rate_1 = 5e-3
        self.learning_rate_2 = 5e-3
        self.learning_rate_3 = 5e-3

        # Our parameters (weights)
        # w1: 3 x 100
        self.w1 = torch.FloatTensor(self.input_dim, self.hidden_dim).uniform_(-3, 3) / self.hidden_dim
        self.w1.requires_grad_()

        # w2: 100 x 100
        self.w2 = torch.FloatTensor(self.hidden_dim, self.hidden_dim).uniform_(-3, 3) / self.hidden_dim
        self.w2.requires_grad_()

        # w3: 100 x 3
        self.w3 = torch.FloatTensor(self.hidden_dim, self.output_dim).uniform_(-3, 3) / self.output_dim
        self.w3.requires_grad_()

        # create infenitisimal generator
        self.gen = InfGenerator()

        # Initialize dynamical system
        self.sys = self.gen.sys

        # Momentum in gradient
        self.gamma = .1
        self.dJ_dw1_m = torch.zeros(self.w1.size())
        self.dJ_dw2_m = torch.zeros(self.w2.size())
        self.dJ_dw3_m = torch.zeros(self.w3.size())


    def sigmoid(self, s):
        return 1 / (1 + torch.exp(-s)) - 0

    def sigmoid_first_derivative(self, s):
        return 1 * s * (1 - s)
    
    def normalize(self, x:torch.Tensor):
        return x / x.norm(dim=1).reshape(x.size(dim=0), 1)
    
    def normalize_(self, x:torch.Tensor):
        return x / x.norm()

    # Forward propagation
    def forward(self, X:torch.Tensor):
        self.q = X

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

        # Third nonlinearity
        self.y6 = self.normalize(self.sigmoid(self.y5)) # N x 3
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
        return y6.T
    
    # Time derivative of forward function
    def d_dt_forward(self, t:torch.Tensor):
        dy5_dy4 = self.w3 # 100 x 3
        dy4_dy3 = self.sigmoid_first_derivative(self.y4) # N x 100
        dy5_dy3 = torch.matmul(dy4_dy3, dy5_dy4) # N x 3
        dy3_dy2 = self.w2 # 100 x 100
        dy2_dy1 = self.sigmoid_first_derivative(self.y2) # N x 100
        dy3_dy1 = torch.matmul(dy2_dy1, dy3_dy2) # N x 100
        dy5_dy1 = torch.matmul(dy5_dy3.T, dy3_dy1) # 3 x 100
        dy5_dq = torch.matmul(self.w1, dy5_dy1.T) # 4 x 3
        dy5_dt = torch.matmul(dy5_dq.T, self.sys.q_dot(t)) # 3 x N
        return dy5_dt
    
    # Time derivative of forward function autograd
    def d_dt_forward_auto(self, t:torch.Tensor) -> torch.Tensor:
        jac = torch.autograd.functional.jacobian(self.forward_, self.q[:,0], create_graph=True)
        df_dt = jac @ self.sys.q_dot(t) # 3 x N
        return df_dt
        # breakpoint()
        # self.y6.T.backward(self.sys.q_dot(t)[1:4], retain_graph=True, create_graph=True)
        # ddt_dy6_dq = self.q.grad
        # df_dt = torch.autograd.grad(self.y5.T, t, torch.ones_like(self.y5.T), retain_graph=True, create_graph=True)
        # return df_dt[0].reshape(4, t.size(dim=2))
    
    # Momentum map
    def J_xi(self, t:torch.Tensor) -> torch.Tensor:
        N = t.size(dim=2)
        return torch.matmul(self.sys.dL_dqdot(t)[1:4].T, self.gen.generator(t).matmul(self.y6.reshape(N,3,1)).reshape(N,3).T).diag(0) # 1 x N

    # Time derivative of momentum map
    def d_dt_J_xi(self, t:torch.Tensor) -> torch.Tensor:
        J_xi = self.J_xi(t)
        d_dt_J_xi = torch.autograd.grad(J_xi, t, torch.ones_like(J_xi), retain_graph=True, create_graph=True)
        # d_dt_J_xi = torch.autograd.functional.jacobian(self.J_xi, t, create_graph=True)
        # breakpoint()
        return d_dt_J_xi[0]
    
    # Loss function
    def J_theta(self, t: torch.Tensor) -> torch.Tensor:

        # data size
        N = t.size(dim=1)

        d_dt_forward = self.d_dt_forward_auto(t) # 3 x N

        d_dt_J_xi = self.d_dt_J_xi(t)

        # nonholonomic momentum cost
        J1 =  (torch.matmul(self.sys.dL_dqdot(t)[1:4].T, d_dt_forward).diag(0) - d_dt_J_xi).norm()
        # J1 = 0
        
        # regularization
        # J2 = - 1 * self.y6.norm()
        J2 = 0

        return J1 + J2

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