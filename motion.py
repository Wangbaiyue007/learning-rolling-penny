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

            # precompute trajectory
            t_pre = torch.arange(0., 20, 0.01)
            N_pre = t_pre.size(dim=0)
            t = t.view(1, N_pre)
            self.theta_pre = self.theta(t_pre)
            self.phi_pre = self.phi(t_pre)
            self.x_pre = self.x(t_pre)
            self.y_pre = self.y(t_pre)

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


        '''
        Hamel's formulation
        '''
        def u_alpha(self, type, t:torch.Tensor) -> torch.Tensor:
            """
            Constraint distribution
            """
            if type == 'SE(2)':
                return torch.tensor([1., 0., self.R*torch.cos(self.phi(t[1])), self.R*torch.sin(self.phi(t[1]))]).reshape(4,1)
        
        def u_sigma_a(self, type, t:torch.Tensor) -> torch.Tensor:
            """
            Lie algebra vector field
            """
            if type == 'SE(2)':
                u2 = torch.tensor([0., 1., -self.y(t[3]), self.x(t[2])]).reshape(4,1)
                u3 = torch.tensor([0., 0., 1., 0.]).reshape(4,1)
                u4 = torch.tensor([0., 0., 0., 1.]).reshape(4,1)
                return torch.cat([u2, u3, u4], axis=1)

        def p(self, type, t:torch.Tensor) -> torch.Tensor:
            """
            Momentum in body frame
            """
            if type == 'SE(2)':
                p2 = self.J * self.phi_dot(t[1]) - self.m * self.y(t[3]) * self.x_dot(t[2]) + self.m * self.x(t[2]) * self.y_dot(t[3])
                p3 = self.m * self.x_dot(t[2])
                p4 = self.m * self.y_dot(t[3])
                p1 = self.I * self.theta_dot(t[0]) + self.R * (torch.cos(self.phi(t[1])) * p3 + torch.sin(self.phi(t[1])) * p4)
                return torch.cat([p1, p2, p3, p4], axis=0)
            
        def p_dot(self, type, t:torch.Tensor) -> torch.Tensor:
            """
            Dynamics based on hamel's equations
            """
            if type == 'SE(2)':
                p2_dot = torch.autograd.grad(self.p(type, t)[1], t, torch.ones_like(self.p(type, t)[1]), retain_graph=True, create_graph=True)[0]
                p3_dot = torch.autograd.grad(self.p(type, t)[2], t, torch.ones_like(self.p(type, t)[2]), retain_graph=True, create_graph=True)[0]
                p4_dot = torch.autograd.grad(self.p(type, t)[3], t, torch.ones_like(self.p(type, t)[3]), retain_graph=True, create_graph=True)[0]
                p1_dot = torch.autograd.grad(self.p(type, t)[0], t, torch.ones_like(self.p(type, t)[0]), retain_graph=True, create_graph=True)[0]
                return torch.cat([p1_dot, p2_dot, p3_dot, p4_dot], axis=0)

        '''
        Utils
        '''
        # Lie bracket of two vector fields, assume g1 and g2 are functions of (theta, phi, x, y)
        def bracket(self, g1: torch.Tensor, g2: torch.Tensor, t:torch.Tensor) -> torch.Tensor:
            u1 = g1[0] * (torch.autograd.grad(g2[0], self.theta_pre)) + g1[1] * (torch.autograd.grad(g2[0], self.phi_pre)) + g1[2] * (torch.autograd.grad(g2[0], self.x_pre)) + g1[3] * (torch.autograd.grad(g2[0], self.y_pre)) \
               - g2[0] * (torch.autograd.grad(g1[0], self.theta_pre)) - g2[1] * (torch.autograd.grad(g1[0], self.phi_pre)) - g2[2] * (torch.autograd.grad(g1[0], self.x_pre)) - g2[3] * (torch.autograd.grad(g1[0], self.y_pre))
            u2 = g1[0] * (torch.autograd.grad(g2[1], self.theta_pre)) + g1[1] * (torch.autograd.grad(g2[1], self.phi_pre)) + g1[2] * (torch.autograd.grad(g2[1], self.x_pre)) + g1[3] * (torch.autograd.grad(g2[1], self.y_pre)) \
               - g2[0] * (torch.autograd.grad(g1[1], self.theta_pre)) - g2[1] * (torch.autograd.grad(g1[1], self.phi_pre)) - g2[2] * (torch.autograd.grad(g1[1], self.x_pre)) - g2[3] * (torch.autograd.grad(g1[1], self.y_pre))
            u3 = g1[0] * (torch.autograd.grad(g2[2], self.theta_pre)) + g1[1] * (torch.autograd.grad(g2[2], self.phi_pre)) + g1[2] * (torch.autograd.grad(g2[2], self.x_pre)) + g1[3] * (torch.autograd.grad(g2[2], self.y_pre)) \
               - g2[0] * (torch.autograd.grad(g1[2], self.theta_pre)) - g2[1] * (torch.autograd.grad(g1[2], self.phi_pre)) - g2[2] * (torch.autograd.grad(g1[2], self.x_pre)) - g2[3] * (torch.autograd.grad(g1[2], self.y_pre))
            u4 = g1[0] * (torch.autograd.grad(g2[3], self.theta_pre)) + g1[1] * (torch.autograd.grad(g2[3], self.phi_pre)) + g1[2] * (torch.autograd.grad(g2[3], self.x_pre)) + g1[3] * (torch.autograd.grad(g2[3], self.y_pre)) \
               - g2[0] * (torch.autograd.grad(g1[3], self.theta_pre)) - g2[1] * (torch.autograd.grad(g1[3], self.phi_pre)) - g2[2] * (torch.autograd.grad(g1[3], self.x_pre)) - g2[3] * (torch.autograd.grad(g1[3], self.y_pre))
            return torch.cat([u1, u2, u3, u4], axis=0)
        
        # compute bracket coefficients with known constraint distribution
        def compute_c(self, type, v: torch.Tensor, t: torch.Tensor):
            if type == 'SE(2)':
                c1 = v[0]
                c2 = v[1]
                c3 = v[2] - self.R * torch.cos(self.phi(t)) * v[0] + self.y(t) * v[1]
                c4 = v[3] - self.R * torch.sin(self.phi(t)) * v[0] - self.x(t) * v[1]
                return torch.cat([c1, c2, c3, c4], axis=0)

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
