import torch

class EquationsOfMotion:

    def __init__(self, 
                 Omega: torch.double = 1,
                 omega: torch.double = 1,
                 R: torch.double = 1,
                 phi_0: torch.double = 0,
                 x_0: torch.double = 0,
                 y_0: torch.double = 0) -> None:

        self.Omega: torch.double = Omega
        self.omega: torch.double = omega
        self.R: torch.double = R
        self.phi_0: torch.double = phi_0
        self.x_0: torch.double = x_0
        self.y_0: torch.double = y_0


    '''
    Equations of motion for training
    '''
    def theta_dot(self, t: torch.double):
        return torch.zeros(1)

    def phi_dot(self, t: torch.double):
        return torch.zeros(1)

    def x_dot(self, t: torch.double):
        return self.Omega*self.R*torch.cos(self.omega*t + self.phi_0)

    def y_dot(self, t: torch.double):
        return self.Omega*self.R*torch.cos(self.omega*t + self.phi_0)

    def theta(self, t: torch.double):
        return self.omega*t + self.phi_0

    def phi(self, t: torch.double):
        return self.Omega*t

    def x(self, t: torch.double):
        return self.Omega/self.omega * self.R * torch.sin(self.omega*t + self.phi_0) + self.x_0

    def y(self, t: torch.double):
        return -self.Omega/self.omega * self.R * torch.cos(self.omega*t + self.phi_0) + self.x_0

