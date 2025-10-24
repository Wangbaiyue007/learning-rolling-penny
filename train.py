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
torch.set_default_device('cuda:0')
torch.set_default_tensor_type('torch.cuda.FloatTensor')
t = torch.arange(0., 20, 0.01).to(device)
opts = {
    'data_size': t.size(0),
    'batch_time': 20,
    'batch_size': 50
}
class Sine(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x)
    
network = torch.nn.Sequential(
            torch.nn.Linear(4, 10),
            Sine(),
            torch.nn.Linear(10, 10),
            Sine(),
            torch.nn.Linear(10, 3)
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
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device), s

def visualize(true_p, pred_p, true_A, pred_A, odefunc, itr):

    ax_traj.cla()
    ax_traj.set_title('p1 to p4 vs t')
    ax_traj.set_xlabel('t')
    ax_traj.set_ylabel('p')
    ax_traj.plot(t.cpu().numpy(), true_p.cpu().numpy()[:, 0], 'g-', label='true p1')
    ax_traj.plot(t.cpu().numpy(), true_p.cpu().numpy()[:, 1], 'b-', label='true p2')
    ax_traj.plot(t.cpu().numpy(), true_p.cpu().numpy()[:, 2], 'r-', label='true p3')
    ax_traj.plot(t.cpu().numpy(), true_p.cpu().numpy()[:, 3], 'y-', label='true p4')
    ax_traj.plot(t.cpu().numpy(), pred_p.cpu().numpy()[:, 0], color='grey', linestyle='--', label='pred p1')
    ax_traj.plot(t.cpu().numpy(), pred_p.cpu().numpy()[:, 1], color='grey', linestyle='--', label='pred p2')
    ax_traj.plot(t.cpu().numpy(), pred_p.cpu().numpy()[:, 2], color='grey', linestyle='--', label='pred p3')
    ax_traj.plot(t.cpu().numpy(), pred_p.cpu().numpy()[:, 3], color='grey', linestyle='--', label='pred p4')
    ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
    # ax_traj.set_ylim(-2, 2)
    ax_traj.legend()

    ax_A.cla()
    ax_A.set_title('A1 to A3 vs t')
    ax_A.set_xlabel('t')
    ax_A.set_ylabel('A')
    ax_A.plot(t.cpu().numpy(), true_A.cpu().numpy()[:, 0], 'g-')
    ax_A.plot(t.cpu().numpy(), true_A.cpu().numpy()[:, 1], 'b-')
    ax_A.plot(t.cpu().numpy(), true_A.cpu().numpy()[:, 2], 'r-')
    ax_A.plot(t.cpu().numpy(), pred_A.cpu().numpy()[:, 0], color='green', linestyle='--')
    ax_A.plot(t.cpu().numpy(), pred_A.cpu().numpy()[:, 1], color='blue', linestyle='--')
    ax_A.plot(t.cpu().numpy(), pred_A.cpu().numpy()[:, 2], color='red', linestyle='--')
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
    dynamics_true = InfGenerator().to(device)
    dynamics_param = InfGenerator(nn=network).to(device)
    optimizer = torch.optim.RMSprop(dynamics_param.parameters(), lr=1e-3)

    # with torch.no_grad():
    p0 = dynamics_true.sys.p()
    true_p = odeint(dynamics_true, p0, t).to(device)
    true_p_dot = torch.zeros((t.size(0), 8)).to(device)
    true_A = torch.zeros((t.size(0), 3)).to(device)
    pred_p = torch.zeros((t.size(0), 4)).to(device)
    for i in range(t.size(0)):
        dynamics_true.sys.evaluate(true_p[i])
        pred_p[i] = dynamics_true.sys.p()[:4]
        true_A[i] = dynamics_true.sys.A
    # visualize(true_p, pred_p, true_A, true_A, dynamics_param, 0)

    for itr in range(0, 2001):
        optimizer.zero_grad()
        batch_p0, batch_t, batch_p, s = get_batch(true_p, opts)
        pred_p = odeint(dynamics_param, batch_p0, batch_t).to(device)
        loss = torch.sum(torch.square(pred_p - batch_p))
        loss.backward()
        optimizer.step()
        print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))

        if itr % 100 == 0:
            with torch.no_grad():
                pred_p = odeint(dynamics_param, p0, t)
                pred_A = dynamics_param.net(true_p[:, 4:]).to(device)
                visualize(true_p, pred_p, true_A, pred_A, dynamics_param, itr)

    breakpoint()