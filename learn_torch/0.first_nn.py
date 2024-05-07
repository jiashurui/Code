import torch.nn as nn
import torch.nn.functional as F
import torch

## 继承nn.Module
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, input):
        # input = F.relu(self.conv1(input))
        # input = F.relu(self.conv2(input))
        input = input + 1
        return input

nn1 = Model()
x = torch.tensor(1.0)
y = nn1(x)
print(x)
print(y)
