import torch
from model import FNN

nn = FNN()
num_epochs = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nn.to(device)

t = torch.arange(0, 10, 0.01)
t = t.view(1, t.size(dim=0))
q = nn.sys.q(t)
breakpoint()

for epoch in range(num_epochs):
    # predictions
    xi = nn.forward(q)

    # loss
    J = torch.norm(nn.J_theta(t))

    # print our mean cross entropy loss
    if epoch % 20 == 0:
        print('Epoch {} | Loss: {}'.format(epoch, J))
    
    # train
    nn.train(q, t)