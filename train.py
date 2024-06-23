import torch
from model import FNN
import matplotlib.pyplot as plt

# Network
nn = FNN(input_dim=4)
num_epochs = 5000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nn.to(device)

# randomize time
t = torch.arange(0., 20, 0.01)
N = t.size(dim=0)
t = t.view(1, N)
t = t.repeat(4,1,1)
# t.requires_grad_()
q = nn.sys.q(t)
# q.requires_grad_()

# vector field before training
xi_0 = nn.forward(q).T
xi_Q_0 = nn.xi_Q(t)
xi_Q_0_np = xi_Q_0.detach().cpu().numpy() 

# training
for epoch in range(num_epochs):

    # predictions
    # xi = nn.forward(q)

    # train
    xi, J = nn.train(q, t)

    # print our mean cross entropy loss
    if epoch % 100 == 0:
        print('Epoch {} | Loss: {}'.format(epoch, J))

# vector field after training
xi_Q = nn.forward(q).T
xi_Q_np = xi_Q.detach().cpu().numpy()
xi = nn.xi(t)
xi_np = xi.detach().cpu().numpy()

# motion from training set
theta = nn.sys.q(t)[0].detach().cpu().numpy()
phi = nn.sys.q(t)[1].detach().cpu().numpy()
x = nn.sys.q(t)[2].detach().cpu().numpy()
y = nn.sys.q(t)[3].detach().cpu().numpy()
t_np = t[0].detach().cpu().numpy().reshape(N)

plt.rcParams['text.usetex'] = True
fig = plt.figure(figsize=plt.figaspect(0.3))

# fig 1
ax = fig.add_subplot(1, 3, 1, projection='3d')
ax.plot(xi_Q_0_np[:,1].reshape(N), xi_Q_0_np[:,2].reshape(N), xi_Q_0_np[:,0].reshape(N))
ax.set_xlabel(r'$\partial / \partial x$', fontsize=15)
ax.set_ylabel(r'$\partial / \partial y$', fontsize=15)
ax.set_zlabel(r'$\partial / \partial\phi$', fontsize=15)
ax.set_title('(a) Vector Field Before Training', fontsize=18)

ax = fig.add_subplot(1, 3, 2, projection='3d')
ax.plot(xi_Q_np[:,1].reshape(N), xi_Q_np[:,2].reshape(N), xi_Q_np[:,0].reshape(N))
ax.set_xlabel(r'$\partial / \partial x$', fontsize=15)
ax.set_ylabel(r'$\partial / \partial y$', fontsize=15)
ax.set_zlabel(r'$\partial / \partial\phi$', fontsize=15)
ax.set_title('(b) Vector Field After Training', fontsize=18)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.5, 1.5)

ax = fig.add_subplot(1, 3, 3, projection='3d')
ax.plot(xi_np[:,1].reshape(N), xi_np[:,2].reshape(N), xi_np[:,0].reshape(N))
ax.set_xlabel(r'$x$', fontsize=15)
ax.set_ylabel(r'$y$', fontsize=15)
ax.set_zlabel(r'$\phi$', fontsize=15)
ax.set_title('(c) Lie Algebra After Training', fontsize=18)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.5, 1.5)

plt.savefig('figs/vfield.png')

# fig 2
fig2 = plt.figure(figsize=plt.figaspect(0.3))
ax2 = fig2.add_subplot(1, 3, 1)
ax2.set_title('(a) First Lie Algebra Element', fontsize=18)
ax2.plot(t_np, xi_np[:,0].reshape(N), label=r'$\xi_1^q$')
ax2.legend(loc='upper right')

ax2 = fig2.add_subplot(1, 3, 2)
ax2.set_xlabel('t', fontsize=15)
ax2.set_title('(b) Second Lie Algebra Element', fontsize=18)
ax2.plot(t_np, xi_np[:,1].reshape(N), label=r'$\xi_2^q$')
ax2.plot(t_np, y, label=r'$y$')
ax2.legend(loc='upper right')

ax2 = fig2.add_subplot(1, 3, 3)
ax2.set_title('(c) Third Lie Algebra Element', fontsize=18)
ax2.plot(t_np, xi_np[:,2].reshape(N), label=r'$\xi_3^q$')
ax2.plot(t_np, x, label=r'$x$')
ax2.legend(loc='upper right')

plt.savefig('figs/LieAlg.png')
plt.show()

breakpoint()