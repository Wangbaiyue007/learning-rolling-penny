import torch
import matplotlib.pyplot as plt 
import numpy as np



class InfGenerator(torch.nn.Module):
    def __init__(self, type = 'SE(2)', nn=None) -> None:
        super(InfGenerator, self).__init__()
        self.type = type
        self.net = nn
        self.sys = self.EquationsOfMotion(nn=self.net)

    def forward(self, t, p):
        self.sys.evaluate(t, network=self.net)
        return self.sys.p_dot(p, self.type)
        
        
    class EquationsOfMotion:

        def __init__(self, 
                    nn: torch.nn.Module = None,
                    Omega: float = 1,
                    omega: float = 3,
                    R: float = 3,
                    phi_0: float = 0,
                    x_0: float = 0,
                    y_0: float = 0,
                    I: float = 1,
                    J: float = 1,
                    m: float = 1,
                    t: torch.Tensor = torch.tensor(0.).to(torch.device("cuda:0"))) -> None:
            
            self.nn = nn

            # system parameters
            self.Omega = Omega
            self.omega = omega
            self.R = R
            self.phi_0 = phi_0
            self.x_0 = x_0
            self.y_0 = y_0
            self.I = I
            self.J = J
            self.m = m

            # state at time t
            self.theta_ = self.theta(t, True)
            self.theta_dot_ = self.theta_dot(t)
            self.phi_ = self.phi(t, True)
            self.phi_dot_ = self.phi_dot(t)
            self.x_ = self.x(t, True)
            self.x_dot_ = self.x_dot(t)
            self.y_ = self.y(t, True)
            self.y_dot_ = self.y_dot(t)
            self.q_ = self.q()

            # coordinates
            self.u1 = self.u_alpha('SE(2)', self.q_)
            self.u2, self.u3, self.u4 = self.u_sigma_a('SE(2)')

            # coefficients
            self.Omega_a_ = self.Omega_a('SE(2)')
            self.c_j_i = self.compute_c_j_i('SE(2)')

        '''
        Equations of motion for training
        '''
        def theta_dot(self, t: torch.Tensor) -> torch.Tensor:
            return self.Omega * torch.ones(1)

        def phi_dot(self, t: torch.Tensor) -> torch.Tensor:
            return self.omega * torch.ones(1)

        def x_dot(self, t: torch.Tensor) -> torch.Tensor:
            return self.Omega*self.R*torch.cos(self.omega*t + self.phi_0)

        def y_dot(self, t: torch.Tensor) -> torch.Tensor:
            return self.Omega*self.R*torch.sin(self.omega*t + self.phi_0)
        
        def q_dot(self, t: torch.Tensor) -> torch.Tensor:
            return torch.cat([self.theta_dot(t[0]), self.phi_dot(t[1]), self.x_dot(t[2]), self.y_dot(t[3])], axis=0)

        def theta(self, t: torch.Tensor, requires_grad: bool = False) -> torch.Tensor:
            theta = self.Omega*t
            if requires_grad:
                theta.requires_grad_()
            return theta

        def phi(self, t: torch.Tensor, requires_grad: bool = False) -> torch.Tensor:
            phi = self.omega*t + self.phi_0
            if requires_grad:
                phi.requires_grad_()
            return phi

        def x(self, t: torch.Tensor, requires_grad: bool = False) -> torch.Tensor:
            x = self.Omega/self.omega * self.R * torch.sin(self.omega*t + self.phi_0) + self.x_0
            if requires_grad:
                x.requires_grad_()
            return x

        def y(self, t: torch.Tensor, requires_grad: bool = False) -> torch.Tensor:
            y = -self.Omega/self.omega * self.R * torch.cos(self.omega*t + self.phi_0) + self.y_0
            if requires_grad:
                y.requires_grad_()
            return y

        def q(self) -> torch.Tensor:
            return torch.stack([self.theta_, self.phi_, self.x_, self.y_])

        def dL_dqdot(self, t:torch.Tensor) -> torch.Tensor:
            return torch.cat([self.I * self.theta_dot(t[0]), self.J * self.phi_dot(t[1]), self.m * self.x_dot(t[2]), self.m * self.y_dot(t[3])], axis=0)


        ## 
        ## Hamel's formulation
        ## 
        def u_alpha(self, q) -> torch.Tensor:
            """
            Constraint distribution. Input a time scalar t.
            """
            phi = self.phi_
            if self.nn is None:
                u1 =  self.R*torch.cos(phi)*torch.tensor([0., 0., 1., 0.]) + \
                    self.R*torch.sin(phi)*torch.tensor([0., 0., 0., 1.]) + \
                    torch.tensor([1., 0., 0., 0.])
            else:
                A_learned = self.nn(q)
                u1 =  torch.tensor([1., 0., 0., 0.], device=A_learned.device) + \
                    - A_learned[0]*torch.tensor([0., 1., 0., 0.], device=A_learned.device) + \
                    - A_learned[1]*torch.tensor([0., 0., 1., 0.], device=A_learned.device) + \
                    - A_learned[2]*torch.tensor([0., 0., 0., 1.], device=A_learned.device)
            return u1
        

        def u_sigma_a(self, q) -> torch.Tensor:
            """
            Lie algebra vector field
            """
            x = q[2]
            y = q[3]
            u2 = y*torch.tensor([0., 0., -1., 0.], device=y.device) + \
                    x*torch.tensor([0., 0., 0., 1.], device=y.device) + \
                    torch.tensor([0., 1., 0., 0.], device=y.device)
            u3 = torch.tensor([0., 0., 1., 0.], device=y.device)
            u4 = torch.tensor([0., 0., 0., 1.], device=y.device)
            return u2, u3, u4

        def Omega_a(self, type) -> torch.Tensor:
            """
            Quasi-velocities
            """
            omega1 = self.phi_dot_
            omega2 = self.u1[1]*self.theta_dot_ + self.phi_dot_
            omega3 = self.u1[2]*self.theta_dot_ + (self.y_*self.phi_dot_ + self.x_dot_)
            omega4 = self.u1[3]*self.theta_dot_ + (-self.x_*self.phi_dot_ + self.y_dot_)
            return torch.cat([omega1, omega2, omega3, omega4])

        def p(self, type) -> torch.Tensor:
            """
            Momentum in body frame
            """
            p2 = self.J * self.phi_dot_ - self.m * self.y_ * self.x_dot_ + self.m * self.x_ * self.y_dot_
            p3 = self.m * self.x_dot_
            p4 = self.m * self.y_dot_
            p1 = self.I * self.theta_dot_ - self.u1[1:4] @ torch.tensor([p2, p3, p4])
            return torch.stack([p1[0], p2[0], p3, p4])
            
        def p_dot(self, p:torch.Tensor, type) -> torch.Tensor:
            """
            Dynamics based on hamel's equations
            """
            c_2_1 = self.c_j_i[0][0]
            c_3_1 = self.c_j_i[0][1]
            c_4_1 = self.c_j_i[0][2]
            c_3_2 = self.c_j_i[1][1]
            c_4_2 = self.c_j_i[1][2]
            c_4_3 = self.c_j_i[2][2]
            c_1_2 = -c_2_1
            c_1_3 = -c_3_1
            c_1_4 = -c_4_1
            c_2_3 = -c_3_2
            c_2_4 = -c_4_2
            c_3_4 = -c_4_3

            p1_dot = (c_2_1 @ p.T) * self.Omega_a_[1] + \
                        (c_3_1 @ p.T) * self.Omega_a_[2] + \
                        (c_4_1 @ p.T) * self.Omega_a_[3]
            p2_dot = (c_1_2 @ p.T) * self.Omega_a_[0] + \
                        (c_3_2 @ p.T) * self.Omega_a_[2] + \
                        (c_4_2 @ p.T) * self.Omega_a_[3]
            p3_dot = (c_1_3 @ p.T) * self.Omega_a_[0] + \
                        (c_2_3 @ p.T) * self.Omega_a_[1] + \
                        (c_4_3 @ p.T) * self.Omega_a_[3]
            p4_dot = (c_1_4 @ p.T) * self.Omega_a_[0] + \
                        (c_2_4 @ p.T) * self.Omega_a_[1] + \
                        (c_3_4 @ p.T) * self.Omega_a_[2]
            return torch.stack([p1_dot, p2_dot, p3_dot, p4_dot]).T

        def evaluate(self, t: torch.Tensor, network: torch.nn.Module=None) -> torch.Tensor:
            # state at time t
            self.theta_ = self.theta(t)
            self.theta_dot_ = self.theta_dot(t)
            self.phi_ = self.phi(t)
            self.phi_dot_ = self.phi_dot(t)
            self.x_ = self.x(t)
            self.x_dot_ = self.x_dot(t)
            self.y_ = self.y(t)
            self.y_dot_ = self.y_dot(t)
            self.q_ = self.q()

            # coordinates
            self.u1 = self.u_alpha('SE(2)', self.q_)
            self.u2, self.u3, self.u4 = self.u_sigma_a('SE(2)')

            # coefficients
            self.Omega_a_ = self.Omega_a('SE(2)')
            self.c_j_i = self.compute_c_j_i('SE(2)')

            return

        ##
        ## Utils
        ##

        def bracket(self, func1, func2, q) -> torch.Tensor:
            """
            Lie bracket of two vector fields, assume g1 and g2 are functions of (theta, phi, x, y)
            """
            coords = (self.theta_, self.phi_, self.x_, self.y_)
            device = coords[0].device

            J_g1 = torch.autograd.functional.jacobian(func1, coords, create_graph=True)
            J_g2 = torch.autograd.functional.jacobian(func2, coords, create_graph=True)

            u1 = func1(q) * J_g2[0] - func2(q) * J_g1[0]
            u2 = func1(q) * J_g2[1] - func2(q) * J_g1[1]
            u3 = func1(q) * J_g2[2] - func2(q) * J_g1[2]
            u4 = func1(q) * J_g2[3] - func2(q) * J_g1[3]
            
            return torch.stack([u1, u2, u3, u4])

        def compute_c(self, type, v: torch.Tensor):
            """
            compute bracket coefficients with known constraint distribution
            """
            if type == 'SE(2)':
                c1 = v[0]
                c2 = v[1] + self.u1[1] * v[0]
                c3 = v[2] + self.u1[2] * v[0] + self.y_ * v[1]
                c4 = v[3] + self.u1[3] * v[0] - self.x_ * v[1]
                return torch.stack([c1, c2, c3, c4])
        
        def compute_c_j_i(self, type):
            c_2_1 = self.compute_c(type, self.bracket(self.u2, self.u1))
            c_3_1 = self.compute_c(type, self.bracket(self.u3, self.u1))
            c_4_1 = self.compute_c(type, self.bracket(self.u4, self.u1))
            c_3_2 = self.compute_c(type, self.bracket(self.u3, self.u2))
            c_4_2 = self.compute_c(type, self.bracket(self.u4, self.u2))
            c_4_3 = self.compute_c(type, self.bracket(self.u4, self.u3))
            c_1_2 = -c_2_1
            c_1_3 = -c_3_1
            c_1_4 = -c_4_1
            c_2_3 = -c_3_2
            c_2_4 = -c_4_2
            c_3_4 = -c_4_3
            c_i_1 = torch.stack([c_2_1, c_3_1, c_4_1], axis=0)
            c_i_2 = torch.stack([c_1_2, c_3_2, c_4_2], axis=0)
            c_i_3 = torch.stack([c_1_3, c_2_3, c_4_3], axis=0)
            c_i_4 = torch.stack([c_1_4, c_2_4, c_3_4], axis=0)
            return torch.stack([c_i_1, c_i_2, c_i_3, c_i_4], axis=0)

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
