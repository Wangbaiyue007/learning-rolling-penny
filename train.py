import torch
from model import FNN
# import torch.optim as optim

nn = FNN()
num_epochs = 101
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nn.to(device)
# optimizer = optim.SGD(nn.parameters(), lr=0.001, momentum=0.9)

t = torch.arange(0, 10, 0.01)
t = t.view(1, t.size(dim=0))
q = nn.sys.q(t)[1:4]

for epoch in range(num_epochs):
    # predictions
    xi = nn.forward(q)

    # train
    xi, J = nn.train(q, t)

    # print our mean cross entropy loss
    if epoch % 1 == 0:
        print('Epoch {} | Loss: {}'.format(epoch, J))
        # breakpoint()
    