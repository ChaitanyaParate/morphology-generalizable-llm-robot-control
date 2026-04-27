import torch
import torch.nn as nn
from torch import optim

class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(1, 1)

d = torch.device('cuda')
agent = Dummy().to(d)
opt = optim.Adam(agent.parameters(), lr=0.1)

agent.to('cpu')
agent.to(d)
opt.step()
print("Opt step executed without crashing?")
