import torch
from model import FNN
from plot import Plot

# Network
nn = FNN(type='S1xR2', input_dim=4)
num_epochs = 3001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nn.to(device)

# initialize time
t = torch.arange(0., 20, 0.01)
N = t.size(dim=0)
t = t.view(1, N)
t = t.repeat(4,1,1)
q = nn.sys.q(t)

# vector field before training
xi_0 = nn.forward(q).T
xi_Q_0 = nn.xi_Q(t)

# training
for epoch in range(num_epochs):

    # train
    xi, J = nn.train(q, t)

    # print our loss value
    if epoch % 100 == 0:
        print('Epoch {} | Loss: {}'.format(epoch, J))

# Plotting the result
P = Plot(nn, t, xi_Q_0)
P.plot(save_fig=True)

breakpoint()