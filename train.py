from datetime import datetime
from motion import InfGenerator
from plot import plot
import torch
from torchdiffeq import odeint
import numpy as np
import os
import matplotlib.pyplot as plt

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

makedirs('png')
makedirs('figs')
makedirs('weights')
fig = plt.figure(figsize=(12, 4), facecolor='white')
ax_traj = fig.add_subplot(121, frameon=False)
ax_A = fig.add_subplot(122, frameon=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device('cuda:0')
torch.set_default_tensor_type('torch.cuda.FloatTensor')
t = torch.arange(0., 20, 0.01).to(device)
opts = {
    'data_size': t.size(0),
    'batch_time': 20,
    'batch_size': 100
}
class Sine(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x)
    
network = torch.nn.Sequential(
            torch.nn.Linear(4, 20),
            Sine(),
            torch.nn.Linear(20, 20),
            Sine(),
            torch.nn.Linear(20, 20),
            torch.nn.ELU(alpha=0.5),
            torch.nn.Linear(20, 3)
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

if __name__ == "__main__":
    dynamics_true = InfGenerator().to(device)
    dynamics_param = InfGenerator(nn=network).to(device)
    optimizer = torch.optim.RMSprop(dynamics_param.parameters(), lr=1e-3)
    torch.save(dynamics_param.state_dict(), 'weights/not_trained_model_{}.pth'.format(datetime.now().strftime("%m_%d-%H:%M:%S")))

    p0 = dynamics_true.sys.p()
    true_p = odeint(dynamics_true, p0, t).to(device)
    true_A = torch.zeros((t.size(0), 3)).to(device)
    for i in range(t.size(0)):
        dynamics_true.sys.evaluate(true_p[i])
        true_A[i] = - dynamics_true.sys.u1[1:4]

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
                plot(true_p, pred_p, true_A, pred_A, ax_traj, ax_A, fig, t, itr, True)
        
    # save trained model
    torch.save(dynamics_param.state_dict(), 'weights/trained_model_{}.pth'.format(datetime.now().strftime("%m_%d-%H:%M:%S")))