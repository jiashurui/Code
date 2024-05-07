import torch
import torch.nn as nn

# L1
a = torch.tensor([1,2,3]).float()
b = torch.tensor([1,2,5]).float()
l1_loss = nn.L1Loss()
print("L1 Loss:" + str(l1_loss(a, b)))

# MSE
mse_loss = nn.MSELoss()
print("MSE Loss :" + str(mse_loss(a, b)))

#