import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Network(nn.Module):

    def __init__(self, D_obs):
        super(Network, self).__init__()
        hid_1 = 170 
        hid_3 = 5
        hid_2 = 29

        self.fc_h1 = nn.Linear(D_obs, hid_1)
        self.fc_h2 = nn.Linear(hid_1, hid_2)
        self.fc_h3 = nn.Linear(hid_2, hid_3)
        self.fc_v  = nn.Linear(hid_3, 1)

    def forward(self, obs):
        h1 = F.tanh(self.fc_h1(obs))
        h2 = F.tanh(self.fc_h2(h1))
        h3 = F.tanh(self.fc_h3(h2))
        v  = self.fc_v(h3)
        return v

for trial in range(50):
    net = Network(17)
    net.load_state_dict(torch.load("init_state.torch"))

    lr = 1e-3
    optim = torch.optim.Adam(
            net.parameters(),
            lr = lr
        )
    batch_sz = 4096
    clip_val = 1.0

    X = Variable(torch.from_numpy(np.load('data.npy')))
    y = Variable(torch.from_numpy(np.load('label.npy')))

    num_batch = int(X.size()[0]/batch_sz)
    for i in range(num_batch - 1):
        X_batch = X[i * batch_sz : (i+1)* batch_sz]
        y_batch = y[i * batch_sz : (i+1)* batch_sz]
        y_pred  = net(X_batch)
        loss = (y_batch - y_pred).pow(2).mean()
        net.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(net.parameters(), clip_val)
        optim.step()


    X_batch = X[(num_batch - 1) * batch_sz : num_batch * batch_sz]
    if np.isnan(np.sum(net(X_batch).data.numpy())):
        print("trial {}: NaN Occurred!".format(trial + 1))
    else: print("trial {}:everything is fine".format(trial + 1))
