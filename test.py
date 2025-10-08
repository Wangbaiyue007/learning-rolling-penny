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
ax_traj = fig.add_subplot(121, frameon=False)
ax_A = fig.add_subplot(122, frameon=False)
# ax_phase = fig.add_subplot(132, frameon=False)
# ax_vecfield = fig.add_subplot(133, frameon=False)
plt.show(block=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
t = torch.arange(0., 20, 0.01).to(device)
opts = {
    'data_size': t.size(0),
    'batch_time': 2,
    'batch_size': 50
}

network = torch.nn.Sequential(
            torch.nn.Linear(4, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(4, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 3),
        ).to(device)
for m in network.modules():
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0, std=0.1)
        torch.nn.init.constant_(m.bias, val=0)

def get_batch(true_y, options):
    s = torch.from_numpy(np.random.choice(np.arange(options['data_size'] - options['batch_time'], dtype=np.int64), options['batch_size'], replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:options['batch_time']]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(options['batch_time'])], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)

def visualize(true_p, pred_p, A, odefunc, itr):

    ax_traj.cla()
    ax_traj.set_title('p1 to p4 vs t')
    ax_traj.set_xlabel('t')
    ax_traj.set_ylabel('p')
    ax_traj.plot(
        t.cpu().numpy(), true_p.cpu().numpy()[:, 0], 'g-', 
        t.cpu().numpy(), true_p.cpu().numpy()[:, 1], 'b-', 
        t.cpu().numpy(), true_p.cpu().numpy()[:, 2], 'r-', 
        t.cpu().numpy(), true_p.cpu().numpy()[:, 3], 'y-')
    ax_traj.plot(t.cpu().numpy(), pred_p.cpu().numpy()[:, 0], color='grey', linestyle='--')
    ax_traj.plot(t.cpu().numpy(), pred_p.cpu().numpy()[:, 1], color='grey', linestyle='--')
    ax_traj.plot(t.cpu().numpy(), pred_p.cpu().numpy()[:, 2], color='grey', linestyle='--')
    ax_traj.plot(t.cpu().numpy(), pred_p.cpu().numpy()[:, 3], color='grey', linestyle='--')
    ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
    # ax_traj.set_ylim(-2, 2)
    ax_traj.legend()

    ax_A.cla()
    ax_A.set_title('A1 to A3 vs t')
    ax_A.set_xlabel('t')
    ax_A.set_ylabel('A')
    ax_A.plot(t.cpu().numpy(), A.detach().numpy()[:, 0], 'g-')
    ax_A.plot(t.cpu().numpy(), A.detach().numpy()[:, 1], 'b-')
    ax_A.plot(t.cpu().numpy(), A.detach().numpy()[:, 2], 'r-')
    ax_A.set_xlim(t.cpu().min(), t.cpu().max())
    ax_A.legend()

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
    plt.draw()
    plt.pause(0.1)

if __name__ == "__main__":
    t = torch.arange(0., 20, 0.01).to(device)
    dynamics_true = InfGenerator().to(device)
    dynamics_param = InfGenerator(nn=network).to(device)
    optimizer = torch.optim.RMSprop(dynamics_param.parameters(), lr=1e-3)

    p0 = dynamics_true.sys.p('SE(2)')
    true_p = odeint(dynamics_true, p0, t)

    for itr in range(1, 11):
        optimizer.zero_grad()
        batch_p0, batch_t, batch_p = get_batch(true_p, opts)
        pred_p = odeint(dynamics_param, batch_p0, batch_t).to(device)
        loss = torch.mean(torch.abs(pred_p - batch_p))
        loss.backward(retain_graph=True)
        optimizer.step()
        print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))

        pred_p = odeint(dynamics_param, p0, t)
        A = dynamics_param.net(true_p)
        with torch.no_grad():
            visualize(true_p, pred_p, A, dynamics_param, itr)