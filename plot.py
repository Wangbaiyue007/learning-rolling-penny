from model import FNN
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.size' : 11,
    'text.usetex': True,
    'pgf.rcfonts': False,
})

class Plot():
    def __init__(self, nn: FNN, t: torch.Tensor, xi_Q_0: torch.Tensor) -> None:
        q = nn.sys.q(t)
        self.N = t.size(dim=2)
        self.gen_type = nn.gen_type

        # vector field before training
        self.xi_Q_0_np = xi_Q_0.detach().cpu().numpy() 

        # vector field after training
        xi_Q = nn.forward(q).T
        self.xi_Q_np = xi_Q.detach().cpu().numpy()
        xi = nn.xi(t)
        self.xi_np = xi.detach().cpu().numpy()

        # motion from training set
        self.theta = q[0].detach().cpu().numpy()
        self.phi = q[1].detach().cpu().numpy()
        self.x = q[2].detach().cpu().numpy()
        self.y = q[3].detach().cpu().numpy()
        self.t = t[0].detach().cpu().numpy().reshape(self.N)
        self.R = nn.sys.R

    def plot(self, save_fig=True) -> None:
        plt.rcParams['text.usetex'] = True
        fig = plt.figure(figsize=plt.figaspect(0.22))

        # fig 1
        ax = fig.add_subplot(1, 3, 1, projection='3d')
        ax.plot(self.xi_Q_0_np[:,1].reshape(self.N), self.xi_Q_0_np[:,2].reshape(self.N), self.xi_Q_0_np[:,0].reshape(self.N), color='0.0')
        ax.set_xlabel(r'$\partial / \partial x$', fontsize=15)
        ax.set_ylabel(r'$\partial / \partial y$', fontsize=15)
        ax.set_zlabel(r'$\partial / \partial\phi$', fontsize=15)
        ax.set_title('(a) Vector Field Before Training', fontsize=18)

        ax = fig.add_subplot(1, 3, 2, projection='3d')
        ax.plot(self.xi_Q_np[:,1].reshape(self.N), self.xi_Q_np[:,2].reshape(self.N), self.xi_Q_np[:,0].reshape(self.N), color='0.0')
        ax.set_xlabel(r'$\partial / \partial x$', fontsize=15)
        ax.set_ylabel(r'$\partial / \partial y$', fontsize=15)
        ax.set_zlabel(r'$\partial / \partial\phi$', fontsize=15)
        ax.set_title('(b) Vector Field After Training', fontsize=18)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)

        ax = fig.add_subplot(1, 3, 3, projection='3d')
        ax.plot(self.xi_np[:,1].reshape(self.N), self.xi_np[:,2].reshape(self.N), self.xi_np[:,0].reshape(self.N), color='0.0')
        ax.set_xlabel(r'$x$', fontsize=15)
        ax.set_ylabel(r'$y$', fontsize=15)
        ax.set_zlabel(r'$\phi$', fontsize=15)
        ax.set_title('(c) Lie Algebra After Training', fontsize=18)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)

        if save_fig: plt.savefig(f'figs/vfield_{self.gen_type}.pgf', bbox_inches='tight')

        # fig 2
        fig2 = plt.figure(figsize=plt.figaspect(0.22))
        # fig2.set_size_inches(w=6.5, h=1.5)
        ax2 = fig2.add_subplot(1, 3, 1)
        ax2.set_title('(a) First Lie Algebra Element', fontsize=18)
        ax2.plot(self.t, self.xi_np[:,0].reshape(self.N), label=r'$\xi_1^q$', color='0.0')
        if self.gen_type == 'S1xR2':
            ax2.plot(self.t, 1 / np.sqrt(self.R**2 + 1) * np.ones_like(self.t), label=r'$\frac{1}{\sqrt{R^2+1}}$', color='0.5', ls='--', linewidth=2)
        ax2.legend(loc='upper right')
        ax2.grid()

        ax2 = fig2.add_subplot(1, 3, 2)
        ax2.set_xlabel('t', fontsize=15)
        ax2.set_title('(b) Second Lie Algebra Element', fontsize=18)
        ax2.plot(self.t, self.xi_np[:,1].reshape(self.N), label=r'$\xi_2^q$', color='0.0')
        if self.gen_type == 'SE(2)':
            ax2.plot(self.t, self.y, label=r'$y$', color='0.5', ls='--', linewidth=2)
        elif self.gen_type == 'S1xR2':
            ax2.plot(self.t, self.R / np.sqrt(self.R**2 + 1) * np.cos(self.phi), label=r'$\frac{R}{\sqrt{R^2+1}}\cos\phi$', color='0.5', ls='--', linewidth=2)
        ax2.legend(loc='upper right')
        ax2.grid()

        ax2 = fig2.add_subplot(1, 3, 3)
        ax2.set_title('(c) Third Lie Algebra Element', fontsize=18)
        ax2.plot(self.t, self.xi_np[:,2].reshape(self.N), label=r'$\xi_3^q$', color='0.0')
        if self.gen_type == 'SE(2)':
            ax2.plot(self.t, self.x, label=r'$x$', color='0.5', ls='--', linewidth=2)
        elif self.gen_type == 'S1xR2':
            ax2.plot(self.t, self.R / np.sqrt(self.R**2 + 1) * np.sin(self.phi), label=r'$\frac{R}{\sqrt{R^2+1}}\sin\phi$', color='0.5', ls='--', linewidth=2)
        ax2.legend(loc='upper right')
        ax2.grid()

        if save_fig: plt.savefig(f'figs/LieAlg_{self.gen_type}.pgf')
        
        plt.show()
