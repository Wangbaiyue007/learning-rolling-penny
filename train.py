import torch
from model import FNN
import matplotlib.pyplot as plt

# Network
nn = FNN(input_dim=4)
num_epochs = 10000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nn.to(device)

# randomize time
t = torch.arange(0., 20., 0.02)
N = t.size(dim=0)
t = t.view(1, N)
t = t.repeat(4,1,1)
t.requires_grad_()
q = nn.sys.q(t)
q.requires_grad_()

# vector field before training
xi_Q_0 = nn.gen.generator(t).matmul(nn.forward(q).T.reshape(N,3,1))
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

        # t = torch.cat([t[10:], t[:10]], dim=0)
        # q = torch.cat([q[:,10:], q[:,:10]], dim=1)

        # randomize initial conditions
        # nn.sys.phi_0 = torch.randn(1)
        # nn.sys.x_0 = torch.randn(1)
        # nn.sys.y_0 = torch.randn(1)
    

# vector field after training
xi = nn.forward(q).T
xi_np = xi.detach().cpu().numpy()
xi_Q = nn.gen.generator(t).matmul(xi.reshape(N,3,1))
xi_Q_np = xi_Q.detach().cpu().numpy() 

print('xi_Q = {}'.format(xi_Q))
print('xi(1, 0, 0) = {}'.format(nn.forward_(torch.tensor([0.,1.,0.,0.]).T)))
print('xi(0, 1, 0) = {}'.format(nn.forward_(torch.tensor([0.,0.,1.,0.]).T)))
print('xi(0, 0, 1) = {}'.format(nn.forward_(torch.tensor([0.,0.,0.,1.]).T)))

plt.rcParams['text.usetex'] = True
fig = plt.figure(figsize=plt.figaspect(0.3))

ax = fig.add_subplot(1, 3, 1, projection='3d')
ax.plot(xi_Q_0_np[:,1].reshape(N), xi_Q_0_np[:,2].reshape(N), xi_Q_0_np[:,0].reshape(N))
ax.set_xlabel(r'$\partial / \partial x$', fontsize=15)
ax.set_ylabel(r'$\partial / \partial y$', fontsize=15)
ax.set_zlabel(r'$\partial / \partial\phi$', fontsize=15)

ax = fig.add_subplot(1, 3, 2, projection='3d')
ax.plot(xi_Q_np[:,1].reshape(N), xi_Q_np[:,2].reshape(N), xi_Q_np[:,0].reshape(N))
ax.set_xlabel(r'$\partial / \partial x$', fontsize=15)
ax.set_ylabel(r'$\partial / \partial y$', fontsize=15)
ax.set_zlabel(r'$\partial / \partial\phi$', fontsize=15)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.5, 1.5)

ax = fig.add_subplot(1, 3, 3, projection='3d')
ax.plot(xi_np[:,1].reshape(N), xi_np[:,2].reshape(N), xi_np[:,0].reshape(N))
ax.set_xlabel(r'$x$', fontsize=15)
ax.set_ylabel(r'$y$', fontsize=15)
ax.set_zlabel(r'$\phi$', fontsize=15)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.5, 1.5)
plt.show()


breakpoint()