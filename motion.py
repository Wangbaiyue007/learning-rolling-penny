import torch
import matplotlib.pyplot as plt 
import numpy as np

class InfGenerator:
    def __init__(self, type = 'SE(2)') -> None:
        self.type = type
        self.sys = self.EquationsOfMotion()

    def generator(self, t:torch.Tensor) -> torch.Tensor:
        N = t.size(dim=2)
        if self.type == 'SE(2)':
            gen = self.sys.y(t[2]).reshape(N,1,1)*torch.tensor([[0., 0., 0.], [-1., 0., 0.], [0., 0., 0.]]) + \
                    self.sys.x(t[3]).reshape(N,1,1)*torch.tensor([[0., 0., 0.], [0., 0., 0.], [1., 0., 0.]]) + \
                    torch.tensor([[1., 0, 0], [0., 1., 0], [0., 0., 1.]]).repeat(N,1,1).reshape(N,3,3)
            return gen
            # return torch.tensor([[1., 0, 0], [-self.sys.y(t), 1., 0], [self.sys.x(t), 0, 1]])
        elif self.type == 'S1xR2':
            return torch.tensor([[1., 0, 0], [0, 1., 0], [0, 0, 1.]]).repeat(N,1,1).reshape(N,3,3)
        
    def generator_inv(self, t: torch.Tensor) -> torch.Tensor:
        N = t.size(dim=2)
        if self.type == 'SE(2)':
            gen = self.sys.y(t[2]).reshape(N,1,1)*torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 0., 0.]]) + \
                    self.sys.x(t[3]).reshape(N,1,1)*torch.tensor([[0., 0., 0.], [0., 0., 0.], [-1., 0., 0.]]) + \
                    torch.tensor([[1., 0, 0], [0., 1., 0], [0., 0., 1.]]).repeat(N,1,1).reshape(N,3,3)
            return gen
            # return torch.tensor([[1., 0, 0], [-self.sys.y(t), 1., 0], [self.sys.x(t), 0, 1]])
        elif self.type == 'S1xR2':
            return torch.tensor([[1., 0, 0], [0, 1., 0], [0, 0, 1.]]).repeat(N,1,1).reshape(N,3,3)
        
    class EquationsOfMotion:

        def __init__(self, 
                    Omega: torch.double = 1,
                    omega: torch.double = 3,
                    R: torch.double = 3,
                    phi_0: torch.double = 0,
                    x_0: torch.double = 0,
                    y_0: torch.double = 0,
                    I: torch.double = 1,
                    J: torch.double = 1,
                    m: torch.double = 1) -> None:

            self.Omega = Omega
            self.omega = omega
            self.R = R
            self.phi_0 = phi_0
            self.x_0 = x_0
            self.y_0 = y_0
            self.I = I
            self.J = J
            self.m = m

        '''
        Equations of motion for training
        '''
        def theta_dot(self, t: torch.Tensor) -> torch.Tensor:
            return self.Omega * torch.ones(1, t.size(dim=1))
            # return torch.autograd.grad(self.theta(t), t, torch.ones_like(self.theta(t)), retain_graph=True, create_graph=True)[0]

        def phi_dot(self, t: torch.Tensor) -> torch.Tensor:
            return self.omega * torch.ones(1, t.size(dim=1))
            # return torch.autograd.grad(self.phi(t), t, torch.ones_like(self.phi(t)), retain_graph=True, create_graph=True)[0]

        def x_dot(self, t: torch.Tensor) -> torch.Tensor:
            return self.Omega*self.R*torch.cos(self.omega*t + self.phi_0)
            # return torch.autograd.grad(self.x(t), t, torch.ones_like(self.x(t)), retain_graph=True, create_graph=True)[0]

        def y_dot(self, t: torch.Tensor) -> torch.Tensor:
            return self.Omega*self.R*torch.sin(self.omega*t + self.phi_0)
            # return torch.autograd.grad(self.y(t), t, torch.ones_like(self.y(t)), retain_graph=True, create_graph=True)[0]
        
        def q_dot(self, t: torch.Tensor) -> torch.Tensor:
            return torch.cat([self.theta_dot(t[0]), self.phi_dot(t[1]), self.x_dot(t[2]), self.y_dot(t[3])], axis=0)

        def theta(self, t: torch.Tensor) -> torch.Tensor:
            return self.Omega*t

        def phi(self, t: torch.Tensor) -> torch.Tensor:
            return self.omega*t + self.phi_0

        def x(self, t: torch.Tensor) -> torch.Tensor:
            return self.Omega/self.omega * self.R * torch.sin(self.omega*t + self.phi_0) + self.x_0

        def y(self, t: torch.Tensor) -> torch.Tensor:
            return -self.Omega/self.omega * self.R * torch.cos(self.omega*t + self.phi_0) + self.y_0
        
        def q(self, t: torch.Tensor) -> torch.Tensor:
            return torch.cat([self.theta(t[0]), self.phi(t[1]), self.x(t[2]), self.y(t[3])], axis=0)

        def dL_dqdot(self, t:torch.Tensor) -> torch.Tensor:
            return torch.cat([self.I * self.theta_dot(t[0]), self.J * self.phi_dot(t[1]), self.m * self.x_dot(t[2]), self.m * self.y_dot(t[3])], axis=0)

        def plot(self) -> None:
            x = torch.arange(-5, 5, 0.1)
            y = torch.arange(-5, 5, 0.1)
            X,Y = np.meshgrid(x,y)
            EX = -(Y - self.x_0) * self.omega
            EY = (X - self.x_0) * self.omega

            # Depict illustration 
            plt.figure(figsize=(10, 10)) 
            plt.streamplot(X, Y, EX, EY, density=1.4, linewidth=None, color='#A23BEC')
            plt.title('Motion on the x-y plane') 
            
            # Show plot with grid 
            plt.grid() 
            plt.show()
