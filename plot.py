import torch
from torchdiffeq import odeint
from motion import InfGenerator
import matplotlib
import matplotlib.pyplot as plt

def plot(true_p, pred_p, true_A, pred_A, ax_traj, ax_A, fig, t, itr, png):
    
    if png:
        ax_traj.cla()
        ax_traj.set_title('p1 to p4 vs t')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('p')
        ax_traj.plot(t.cpu().numpy(), true_p.cpu().numpy()[:, 0], color='0.0', linestyle='-')
        ax_traj.plot(t.cpu().numpy(), true_p.cpu().numpy()[:, 1], color='0.0', linestyle='-')
        ax_traj.plot(t.cpu().numpy(), true_p.cpu().numpy()[:, 2], color='0.0', linestyle='-')
        ax_traj.plot(t.cpu().numpy(), true_p.cpu().numpy()[:, 3], color='0.0', linestyle='-')
        ax_traj.plot(t.cpu().numpy(), pred_p.cpu().numpy()[:, 0], color='0.5', linestyle='--')
        ax_traj.plot(t.cpu().numpy(), pred_p.cpu().numpy()[:, 1], color='0.5', linestyle='--')
        ax_traj.plot(t.cpu().numpy(), pred_p.cpu().numpy()[:, 2], color='0.5', linestyle='--')
        ax_traj.plot(t.cpu().numpy(), pred_p.cpu().numpy()[:, 3], color='0.5', linestyle='--')
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())

        ax_A.cla()
        ax_A.set_title('A1 to A3 vs t')
        ax_A.set_xlabel('t')
        ax_A.set_ylabel('A')
        ax_A.plot(t.cpu().numpy(), true_A.cpu().numpy()[:, 0], color='0.0', linestyle='-')
        ax_A.plot(t.cpu().numpy(), true_A.cpu().numpy()[:, 1], color='0.0', linestyle='-')
        ax_A.plot(t.cpu().numpy(), true_A.cpu().numpy()[:, 2], color='0.0', linestyle='-')
        ax_A.plot(t.cpu().numpy(), pred_A.cpu().numpy()[:, 0], color='0.5', linestyle='--')
        ax_A.plot(t.cpu().numpy(), pred_A.cpu().numpy()[:, 1], color='0.5', linestyle='--')
        ax_A.plot(t.cpu().numpy(), pred_A.cpu().numpy()[:, 2], color='0.5', linestyle='--')
        ax_A.set_xlim(t.cpu().min(), t.cpu().max())
        fig.tight_layout()

        plt.savefig('png/{:03d}.png'.format(itr))
        plt.draw()
        plt.pause(0.1)
    else:
        ax_traj.cla()
        ax_traj.set_title(r'$p_1$ to $p_4$ vs $t$', fontsize=18)
        ax_traj.set_xlabel(r'$t$')
        ax_traj.set_ylabel(r'$p$')
        ax_traj.plot(t.cpu().numpy(), true_p.cpu().numpy()[:, 0], color='0.0', linestyle='-')
        ax_traj.plot(t.cpu().numpy(), true_p.cpu().numpy()[:, 1], color='0.0', linestyle='-')
        ax_traj.plot(t.cpu().numpy(), true_p.cpu().numpy()[:, 2], color='0.0', linestyle='-')
        ax_traj.plot(t.cpu().numpy(), true_p.cpu().numpy()[:, 3], color='0.0', linestyle='-')
        ax_traj.plot(t.cpu().numpy(), pred_p.cpu().numpy()[:, 0], color='0.5', linestyle='--', linewidth=2)
        ax_traj.plot(t.cpu().numpy(), pred_p.cpu().numpy()[:, 1], color='0.5', linestyle='--', linewidth=2)
        ax_traj.plot(t.cpu().numpy(), pred_p.cpu().numpy()[:, 2], color='0.5', linestyle='--', linewidth=2)
        ax_traj.plot(t.cpu().numpy(), pred_p.cpu().numpy()[:, 3], color='0.5', linestyle='--', linewidth=2)
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())

        ax_A.cla()
        ax_A.set_title(r'$\mathcal{A}_1$ to $\mathcal{A}_3$ vs $t$', fontsize=18)
        ax_A.set_xlabel(r'$t$')
        ax_A.set_ylabel(r'$\mathcal{A}$')
        ax_A.plot(t.cpu().numpy(), true_A.cpu().numpy()[:, 0], color='0.0', linestyle='-')
        ax_A.plot(t.cpu().numpy(), true_A.cpu().numpy()[:, 1], color='0.0', linestyle='-')
        ax_A.plot(t.cpu().numpy(), true_A.cpu().numpy()[:, 2], color='0.0', linestyle='-')
        ax_A.plot(t.cpu().numpy(), pred_A.cpu().numpy()[:, 0], color='0.5', linestyle='--', linewidth=2)
        ax_A.plot(t.cpu().numpy(), pred_A.cpu().numpy()[:, 1], color='0.5', linestyle='--', linewidth=2)
        ax_A.plot(t.cpu().numpy(), pred_A.cpu().numpy()[:, 2], color='0.5', linestyle='--', linewidth=2)
        ax_A.set_xlim(t.cpu().min(), t.cpu().max())
        fig.tight_layout()

        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'font.size' : 11,
            'text.usetex': True,
            'pgf.rcfonts': False,
        })
        plt.savefig('figs/{:03d}.pgf'.format(itr), bbox_inches='tight')

class Sine(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device('cuda:0')

# load model
weight_path = 'weights/trained_model_10_26-12:57:03.pth'
network = torch.nn.Sequential(
            torch.nn.Linear(4, 20),
            Sine(),
            torch.nn.Linear(20, 20),
            Sine(),
            torch.nn.Linear(20, 20),
            torch.nn.ELU(alpha=0.5),
            torch.nn.Linear(20, 3)
        ).to(device)
dynamics_true = InfGenerator().to(device)
dynamics_param = InfGenerator(nn=network).to(device)
dynamics_param.load_state_dict(torch.load(weight_path, weights_only=True))

# prepare data
t = torch.arange(0., 20, 0.01).to(device)
p0 = dynamics_true.sys.p()
true_p = odeint(dynamics_true, p0, t).to(device)
true_A = torch.zeros((t.size(0), 3)).to(device)
for i in range(t.size(0)):
    dynamics_true.sys.evaluate(true_p[i])
    true_A[i] = - dynamics_true.sys.u1[1:4]
pred_p = odeint(dynamics_param, p0, t)
pred_A = dynamics_param.net(true_p[:, 4:]).to(device)

# plot
fig = plt.figure(figsize=(12, 4))
ax_traj = fig.add_subplot(121, frameon=False)
ax_A = fig.add_subplot(122, frameon=False)
with torch.no_grad():
    plot(true_p, pred_p, true_A, pred_A, ax_traj, ax_A, fig, t, 2000, False)