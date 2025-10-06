from motion import InfGenerator
import torch
from torchdiffeq import odeint
import numpy as np
import os
import matplotlib.pyplot as plt

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

makedirs('png')
fig = plt.figure(figsize=(12, 4), facecolor='white')
ax_traj = fig.add_subplot(111, frameon=False)
# ax_phase = fig.add_subplot(132, frameon=False)
# ax_vecfield = fig.add_subplot(133, frameon=False)
plt.show(block=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
t = torch.arange(0., 20, 0.01).to(device)

def visualize(true_y, odefunc, itr):

    ax_traj.cla()
    ax_traj.set_title('p1 to p4 vs t')
    ax_traj.set_xlabel('t')
    ax_traj.set_ylabel('p')
    ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0], t.cpu().numpy(), true_y.cpu().numpy()[:, 1], t.cpu().numpy(), true_y.cpu().numpy()[:, 2], t.cpu().numpy(), true_y.cpu().numpy()[:, 3], 'g-')
    # ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], 'b--')
    ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
    # ax_traj.set_ylim(-2, 2)
    ax_traj.legend()

    # ax_phase.cla()
    # ax_phase.set_title('Phase Portrait')
    # ax_phase.set_xlabel('x')
    # ax_phase.set_ylabel('y')
    # ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
    # # ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')
    # ax_phase.set_xlim(-2, 2)
    # ax_phase.set_ylim(-2, 2)

    # ax_vecfield.cla()
    # ax_vecfield.set_title('Learned Vector Field')
    # ax_vecfield.set_xlabel('x')
    # ax_vecfield.set_ylabel('y')

    # y, x = np.mgrid[-2:2:21j, -2:2:21j]
    # dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
    # mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
    # dydt = (dydt / mag)
    # dydt = dydt.reshape(21, 21, 2)

    # ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
    # ax_vecfield.set_xlim(-2, 2)
    # ax_vecfield.set_ylim(-2, 2)

    fig.tight_layout()
    plt.savefig('png/{:03d}'.format(itr))
    # plt.draw()
    # plt.pause(0.001)
    plt.show()

if __name__ == "__main__":
    t = torch.arange(0., 20, 0.01).to(device)
    inf_gen = InfGenerator().to(device)
    p0 = inf_gen.sys.p('SE(2)')
    true_p = odeint(inf_gen, p0, t)
    visualize(true_p, inf_gen, 0)