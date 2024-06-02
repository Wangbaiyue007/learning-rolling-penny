import torch
from model import FNN
import matplotlib.pyplot as plt
# import torch.optim as optim

def normalize(x:torch.Tensor):
    return x/x.norm()

# Network
nn = FNN(input_dim=4)
num_epochs = 1000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nn.to(device)

# randomize time
# t = torch.FloatTensor(1, 1000).uniform_(0, 10)
t = torch.arange(0., 20., 0.01)
N = t.size(dim=0)
t = t.view(1, N)
t = t.repeat(4,1,1)
t.requires_grad_()
q = nn.sys.q(t)
q.requires_grad_()
q.retain_grad() # retain grad for non-leaf elements

J_ = 999.

for epoch in range(num_epochs):

    # predictions
    # xi = nn.forward(q)

    # train
    xi, J = nn.train(q, t)

    # print our mean cross entropy loss
    if epoch % 5 == 0:
        print('Epoch {} | Loss: {}'.format(epoch, J))

        # t = torch.cat([t[10:], t[:10]], dim=0)
        # q = torch.cat([q[:,10:], q[:,:10]], dim=1)

        # randomize initial conditions
        # nn.sys.phi_0 = torch.randn(1)
        # nn.sys.x_0 = torch.randn(1)
        # nn.sys.y_0 = torch.randn(1)

        # if J > J_: break
        # breakpoint()
    
    J_ = J


xi_Q = nn.gen.generator(t).matmul(nn.forward(nn.sys.q(t)).T.reshape(N,3,1))
xi_Q_np = xi_Q.detach().cpu().numpy() 

print('xi_Q = {}'.format(xi_Q))
print('xi(1, 0, 0) = {}'.format(normalize(nn.forward(torch.tensor([0.,1.,0.,0.]).T))))
print('xi(0, 1, 0) = {}'.format(normalize(nn.forward(torch.tensor([0.,0.,1.,0.]).T))))
print('xi(0, 0, 1) = {}'.format(normalize(nn.forward(torch.tensor([0.,0.,0.,1.]).T))))

ax = plt.figure().add_subplot(projection='3d')
ax.plot(xi_Q_np[:,1].reshape(N), xi_Q_np[:,2].reshape(N), xi_Q_np[:,0].reshape(N))
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("phi")
plt.show()

breakpoint()