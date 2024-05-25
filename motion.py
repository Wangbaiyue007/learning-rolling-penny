import torch
import matplotlib.pyplot as plt 
import numpy as np

class InfGenerator:
    def __init__(self, type = 'SE(2)') -> None:
        self.type = type
        self.sys = self.EquationsOfMotion()

    def generator(self, t):
        if self.type == 'SE(2)':
            return torch.tensor([[1., 0, 0], [-self.sys.y(t), 1., 0], [self.sys.x(t), 0, 1]])
        elif self.type == 'S1xR2':
            return torch.tensor([[1., 0, 0], [0, 1., 0], [0, 0, 1.]])
        
    def d_dt_generator(self, t):
        if self.type == 'SE(2)':
            return torch.tensor([[0, 0, 0], [-self.sys.y_dot(t), 0, 0], [self.sys.x_dot(t), 0, 0]])
        elif self.type == 'S1xR2':
            return torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        
    class EquationsOfMotion:

        def __init__(self, 
                    Omega: torch.double = 1,
                    omega: torch.double = 1,
                    R: torch.double = 1,
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
        def theta_ddot(self, t: torch.Tensor):
            return torch.zeros(1, t.size(dim=0))

        def phi_ddot(self, t: torch.Tensor):
            return torch.zeros(1, t.size(dim=0))

        def x_ddot(self, t: torch.Tensor):
            return -self.omega*self.Omega*self.R*torch.sin(self.omega*t + self.phi_0)

        def y_ddot(self, t: torch.Tensor):
            return self.omega*self.Omega*self.R*torch.cos(self.omega*t + self.phi_0)
        
        def theta_dot(self, t: torch.Tensor):
            return self.omega * torch.ones(1, t.size(dim=0))

        def phi_dot(self, t: torch.Tensor):
            return self.Omega * torch.ones(1, t.size(dim=0))

        def x_dot(self, t: torch.Tensor):
            return self.Omega*self.R*torch.cos(self.omega*t + self.phi_0)

        def y_dot(self, t: torch.Tensor):
            return self.Omega*self.R*torch.sin(self.omega*t + self.phi_0)
        
        def q_dot(self, t: torch.Tensor):
            return torch.cat([self.theta_dot(t), self.phi_dot(t), self.x_dot(t), self.y_dot(t)], axis=0)

        def theta(self, t: torch.Tensor):
            return self.omega*t

        def phi(self, t: torch.Tensor):
            return self.Omega*t + self.phi_0

        def x(self, t: torch.Tensor):
            return self.Omega/self.omega * self.R * torch.sin(self.omega*t + self.phi_0) + self.x_0

        def y(self, t: torch.Tensor):
            return -self.Omega/self.omega * self.R * torch.cos(self.omega*t + self.phi_0) + self.x_0
        
        def q(self, t: torch.Tensor):
            return torch.cat([self.theta(t), self.phi(t), self.x(t), self.y(t)], axis=0)

        def dL_dqdot(self, t:torch.Tensor):
            return torch.cat([self.I * self.theta_dot(t), self.J * self.phi_dot(t), self.m * self.x_dot(t), self.m * self.y_dot(t)], axis=0)

        def d_dt_dL_dqdot(self, t:torch.Tensor):
            return torch.cat([self.I * self.theta_ddot(t), self.J * self.phi_ddot(t), self.m * self.x_ddot(t), self.m * self.y_ddot(t)], axis=0)
        
        def plot(self):
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
