import torch
from model import FNN
import matplotlib.pyplot as plt
# import torch.optim as optim

def normalize(x:torch.Tensor):
    return x/x.sum(0).expand_as(x).abs()

# Network
nn = FNN()
num_epochs = 101
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nn.to(device)

# randomize time
# t = torch.FloatTensor(1, 1000).uniform_(0, 10)
t = torch.arange(0, 20, 0.01)
N = t.size(dim=0)
t = t.view(1, N)
t.requires_grad_()
q = nn.sys.q(t)[1:4]
q.requires_grad_()
q.retain_grad() # retain grad for non-leaf elements

J_ = 999.

for epoch in range(num_epochs):

    # predictions
    # xi = nn.forward(q)

    # train
    xi, J = nn.train(q, t)

    # print our mean cross entropy loss
    if epoch % 1 == 0:
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

    # if nn.q.grad is not None: nn.q.grad.zero_()
    # if t.grad is not None: t.grad.zero_()

xi_Q = nn.gen.generator(t).matmul(nn.forward(nn.sys.q(t)[1:4]).T.reshape(N,3,1))
xi_Q_np = xi_Q.detach().cpu().numpy() 

print('xi_Q = {}'.format(xi_Q))
print('xi(1, 0, 0) = {}'.format(normalize(nn.forward(torch.tensor([1.,0.,0.]).reshape(3,1)))))
print('xi(0, 1, 0) = {}'.format(normalize(nn.forward(torch.tensor([0.,1.,0.]).reshape(3,1)))))
print('xi(0, 0, 1) = {}'.format(normalize(nn.forward(torch.tensor([0.,0.,1.]).reshape(3,1)))))

ax = plt.figure().add_subplot(projection='3d')
ax.plot(xi_Q_np[:,1].reshape(N), xi_Q_np[:,2].reshape(N), xi_Q_np[:,0].reshape(N))
plt.show()

breakpoint()