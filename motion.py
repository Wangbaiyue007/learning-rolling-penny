import torch
from torch.func import jacrev, vmap
import matplotlib.pyplot as plt 
import numpy as np



class InfGenerator(torch.nn.Module):
    def __init__(self, type = 'SE(2)', nn=None) -> None:
        super(InfGenerator, self).__init__()
        self.type = type
        self.net = nn
        self.sys = self.EquationsOfMotion(nn=self.net)

    def forward(self, t, p):
        self.sys.evaluate(p)
        return self.sys.p_dot(p)
        
        
    class EquationsOfMotion:

        def __init__(self, 
                    nn: torch.nn.Module = None,
                    Omega: float = 1,
                    omega: float = 0.5,
                    R: float = 0.3,
                    phi_0: float = 0,
                    x_0: float = 3,
                    y_0: float = -3,
                    I: float = 1,
                    J: float = 1,
                    m: float = 1,
                    t: torch.Tensor = torch.tensor(0.).to(torch.device("cuda:0"))) -> None:
            
            self.nn = nn if nn is not None else lambda q: torch.stack([torch.tensor(0.).to(q.device), -R*torch.cos(q[1]), -R*torch.sin(q[1])])

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

            # initialize state
            self.theta_ = self.theta(t)
            self.theta_dot_ = self.theta_dot(t)
            self.phi_ = self.phi(t)
            self.phi_dot_ = self.phi_dot(t)
            self.x_ = self.x(t)
            self.x_dot_ = self.x_dot(t)
            self.y_ = self.y(t)
            self.y_dot_ = self.y_dot(t)
            self.q_ = self.q()
            self.xi_ = self.xi_a()

            # coordinates
            self.u1 = self.u_alpha(self.q_)

            # constraint
            self.A = torch.stack([-self.u1.T[1], -(self.u1.T[2] + self.u1.T[1] * self.y_), -(self.u1.T[3] - self.u1.T[1] * self.x_)])

            # coefficients
            self.v_a_ = self.v_a()
            self.c_j_i = self.compute_c_j_i()

        '''
        Equations of motion for training
        '''
        def theta_dot(self, t: torch.Tensor) -> torch.Tensor:
            return self.Omega * torch.ones(1)[0]

        def phi_dot(self, t: torch.Tensor) -> torch.Tensor:
            return self.omega * torch.ones(1)[0]

        def x_dot(self, t: torch.Tensor) -> torch.Tensor:
            return self.Omega*self.R*torch.cos(self.omega*t + self.phi_0)

        def y_dot(self, t: torch.Tensor) -> torch.Tensor:
            return self.Omega*self.R*torch.sin(self.omega*t + self.phi_0)
        
        def q_dot(self, t: torch.Tensor) -> torch.Tensor:
            return torch.cat([self.theta_dot(t[0]), self.phi_dot(t[1]), self.x_dot(t[2]), self.y_dot(t[3])], axis=0)

        def theta(self, t: torch.Tensor) -> torch.Tensor:
            theta = self.Omega*t
            return theta

        def phi(self, t: torch.Tensor) -> torch.Tensor:
            phi = self.omega*t + self.phi_0
            return phi

        def x(self, t: torch.Tensor) -> torch.Tensor:
            x = self.Omega/self.omega * self.R * torch.sin(self.omega*t + self.phi_0) + self.x_0
            return x

        def y(self, t: torch.Tensor) -> torch.Tensor:
            y = -self.Omega/self.omega * self.R * torch.cos(self.omega*t + self.phi_0) + self.y_0
            return y

        def q(self) -> torch.Tensor:
            return torch.stack([self.theta_, self.phi_, self.x_, self.y_])

        ## 
        ## Hamel's formulation
        ## 
        def u_alpha(self, q) -> torch.Tensor:
            """
            Constraint distribution. Input a time scalar t.
            """
            A = self.nn(q).T
            if len(q.shape) > 1:
                u1 =  torch.tensor([1., 0., 0., 0.], device=A.device).repeat(A.size(1), 1).mT + \
                    - A[0]*torch.tensor([0., 1., 0., 0.], device=A.device).repeat(A.size(1), 1).mT + \
                    - A[0]*q[:,3]*torch.tensor([0., 0., -1., 0.], device=A.device).repeat(A.size(1), 1).mT + \
                    - A[0]*q[:,2]*torch.tensor([0., 0., 0., 1.], device=A.device).repeat(A.size(1), 1).mT + \
                    - A[1]*torch.tensor([0., 0., 1., 0.], device=A.device).repeat(A.size(1), 1).mT + \
                    - A[2]*torch.tensor([0., 0., 0., 1.], device=A.device).repeat(A.size(1), 1).mT
                u1 = u1.mT
            else:
                u1 =  torch.tensor([1., 0., 0., 0.], device=A.device) + \
                    - A[0]*torch.tensor([0., 1., 0., 0.], device=A.device) + \
                    - A[0]*q[3]*torch.tensor([0., 0., -1., 0.], device=A.device) + \
                    - A[0]*q[2]*torch.tensor([0., 0., 0., 1.], device=A.device) + \
                    - A[1]*torch.tensor([0., 0., 1., 0.], device=A.device) + \
                    - A[2]*torch.tensor([0., 0., 0., 1.], device=A.device)
            return u1 # n x 4
        

        def u_sigma_a(self, q) -> torch.Tensor:
            """
            Lie algebra vector field
            """
            u2 = q[3]*torch.tensor([0., 0., -1., 0.], device=q.device) + \
                    q[2]*torch.tensor([0., 0., 0., 1.], device=q.device) + \
                    torch.tensor([0., 1., 0., 0.], device=q.device)
            u3 = torch.tensor([0., 0., 1., 0.], device=q.device)
            u4 = torch.tensor([0., 0., 0., 1.], device=q.device)
            return u2, u3, u4
        
        def u_sigma_1(self, q) -> torch.Tensor:
            """
            Lie algebra vector field first element
            """
            if len(q.shape) > 1:
                u = q[:, 3]*torch.tensor([0., 0., -1., 0.], device=q.device).repeat(q.size(0), 1).mT + \
                        q[:, 2]*torch.tensor([0., 0., 0., 1.], device=q.device).repeat(q.size(0), 1).mT + \
                        torch.tensor([0., 1., 0., 0.], device=q.device).repeat(q.size(0), 1).mT
                u = u.mT
            else:
                u = q[3]*torch.tensor([0., 0., -1., 0.], device=q.device) + \
                        q[2]*torch.tensor([0., 0., 0., 1.], device=q.device) + \
                        torch.tensor([0., 1., 0., 0.], device=q.device)
            return u
        
        def u_sigma_2(self, q) -> torch.Tensor:
            """
            Lie algebra vector field second element
            """
            if len(q.shape) > 1:
                u = torch.tensor([0., 0., 1., 0.], device=q.device).repeat(q.size(0), 1)
            else:
                u = torch.tensor([0., 0., 1., 0.], device=q.device)
            return u
        
        def u_sigma_3(self, q) -> torch.Tensor:
            """
            Lie algebra vector field third element
            """
            if len(q.shape) > 1:
                u = torch.tensor([0., 0., 0., 1.], device=q.device).repeat(q.size(0), 1)
            else:
                u = torch.tensor([0., 0., 0., 1.], device=q.device)
            return u
        
        def xi_a(self) -> torch.Tensor:
            """
            Lie algebra elements
            """
            xi2 = self.phi_dot_
            xi3 = self.x_dot_ + self.y_*xi2
            xi4 = self.y_dot_ - self.x_*xi2
            return torch.stack([xi2, xi3, xi4]).T

        def v_a(self) -> torch.Tensor:
            """
            Quasi-velocities
            """
            r_dot = self.theta_dot_
            omega1 = self.xi_.T[0] + self.A[0]*self.theta_dot_
            omega2 = self.xi_.T[1] + self.A[1]*self.theta_dot_
            omega3 = self.xi_.T[2] + self.A[2]*self.theta_dot_
            return torch.stack([r_dot, omega1, omega2, omega3])

        def p(self) -> torch.Tensor:
            """
            Momentum in body frame
            """
            x_dot = -self.y_*self.xi_[0] + self.xi_[1]
            y_dot = self.x_*self.xi_[0] + self.xi_[2]
            p2 = self.J * self.phi_dot_ - self.m * self.y_ * x_dot + self.m * self.x_ * y_dot
            p3 = self.m * x_dot
            p4 = self.m * y_dot
            p1 = self.I * self.theta_dot_ - torch.linalg.vecdot(self.A.T, torch.stack([p2, p3, p4]).T)
            return torch.stack([p1, p2, p3, p4, self.theta_, self.phi_, self.x_, self.y_]).T

        def p_dot(self, p:torch.Tensor) -> torch.Tensor:
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

            p1_dot = torch.linalg.vecdot(c_2_1.T, p.T[:4].T) * self.v_a_[1] + \
                        torch.linalg.vecdot(c_3_1.T, p.T[:4].T) * self.v_a_[2] + \
                        torch.linalg.vecdot(c_4_1.T, p.T[:4].T) * self.v_a_[3]
            p2_dot = torch.linalg.vecdot(c_1_2.T, p.T[:4].T) * self.v_a_[0] + \
                    torch.linalg.vecdot(c_3_2.T, p.T[:4].T) * self.v_a_[2] + \
                    torch.linalg.vecdot(c_4_2.T, p.T[:4].T) * self.v_a_[3]
            p3_dot = torch.linalg.vecdot(c_1_3.T, p.T[:4].T) * self.v_a_[0] + \
                        torch.linalg.vecdot(c_2_3.T, p.T[:4].T) * self.v_a_[1] + \
                        torch.linalg.vecdot(c_4_3.T, p.T[:4].T) * self.v_a_[3]
            p4_dot = torch.linalg.vecdot(c_1_4.T, p.T[:4].T) * self.v_a_[0] + \
                        torch.linalg.vecdot(c_2_4.T, p.T[:4].T) * self.v_a_[1] + \
                        torch.linalg.vecdot(c_3_4.T, p.T[:4].T) * self.v_a_[2]
            return torch.stack([p1_dot, p2_dot, p3_dot, p4_dot, self.theta_dot_, self.phi_dot_, self.x_dot_, self.y_dot_]).T

        def evaluate_q(self, p: torch.Tensor) -> torch.Tensor:
            # state at time t
            p = p.T
            self.theta_ = p[4]
            self.phi_ = p[5]
            self.x_ = p[6]
            self.y_ = p[7]

            self.q_ = self.q().T
            return
        
        def evaluate_state(self, p: torch.Tensor) -> torch.Tensor:
            # state at time t
            p = p.T
            self.theta_dot_ = (p[0] + torch.linalg.vecdot(self.A.T, p[1:4].T))/self.I
            self.x_dot_ = p[2]/self.m
            self.y_dot_ = p[3]/self.m
            self.phi_dot_ = (p[1] + self.y_*p[2] - self.x_*p[3])/self.J

            return
        
        def evaluate(self, p: torch.Tensor) -> torch.Tensor:
            
            # configuration
            self.evaluate_q(p)

            # coordinates
            self.u1 = self.u_alpha(self.q_)

            # constraint
            self.A = torch.stack([-self.u1.T[1], -(self.u1.T[2] + self.u1.T[1] * self.y_), -(self.u1.T[3] - self.u1.T[1] * self.x_)])

            # state at time t
            self.evaluate_state(p)
            self.xi_ = self.xi_a()

            # coefficients
            self.v_a_ = self.v_a()
            self.c_j_i = self.compute_c_j_i()

            return

        ##
        ## Utils
        ##

        def bracket(self, func1, func2, q) -> torch.Tensor:
            """
            Lie bracket of two vector fields, assume g1 and g2 are functions of (theta, phi, x, y)
            """
            v1 = func1(q)
            v2 = func2(q)

            if len(q.shape) <= 1:
                J_g1 = torch.autograd.functional.jacobian(func1, q)
                J_g2 = torch.autograd.functional.jacobian(func2, q)

                u1 = v1 @ J_g2[0] - v2 @ J_g1[0]
                u2 = v1 @ J_g2[1] - v2 @ J_g1[1]
                u3 = v1 @ J_g2[2] - v2 @ J_g1[2]
                u4 = v1 @ J_g2[3] - v2 @ J_g1[3]
            else:
                J_g1 = vmap(jacrev(func1))(q)
                J_g2 = vmap(jacrev(func2))(q)

                u1 = torch.linalg.vecdot(v1, J_g2[:,0,:]) - torch.linalg.vecdot(v2, J_g1[:,0,:])
                u2 = torch.linalg.vecdot(v1, J_g2[:,1,:]) - torch.linalg.vecdot(v2, J_g1[:,1,:])
                u3 = torch.linalg.vecdot(v1, J_g2[:,2,:]) - torch.linalg.vecdot(v2, J_g1[:,2,:])
                u4 = torch.linalg.vecdot(v1, J_g2[:,3,:]) - torch.linalg.vecdot(v2, J_g1[:,3,:])

            return torch.stack([u1, u2, u3, u4]) # n x 4

        def compute_c(self, v: torch.Tensor):
            """
            compute bracket coefficients with known constraint distribution
            """
            k1 = v[0]
            k2 = v[1] + self.A[0] * k1
            k3 = v[2] + self.A[1] * k1 - self.A[0] * self.y_ * k1 + self.y_ * k2
            k4 = v[3] + self.A[2] * k1 - self.A[0] * self.x_ * k1 - self.x_ * k2
            return torch.tensor(torch.stack([k1, k2, k3, k4]))

        def compute_c_j_i(self):
            c_2_1 = self.compute_c(self.bracket(self.u_sigma_1, self.u_alpha, self.q_))
            c_3_1 = self.compute_c(self.bracket(self.u_sigma_2, self.u_alpha, self.q_))
            c_4_1 = self.compute_c(self.bracket(self.u_sigma_3, self.u_alpha, self.q_))
            c_3_2 = self.compute_c(self.bracket(self.u_sigma_2, self.u_sigma_1, self.q_))
            c_4_2 = self.compute_c(self.bracket(self.u_sigma_3, self.u_sigma_1, self.q_))
            if len(c_2_1.shape) > 1:
                c_4_3 = torch.zeros((4, c_2_1.size(1)), device=torch.device("cuda:0"))
            else:
                c_4_3 = torch.zeros(4, device=torch.device("cuda:0"))
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