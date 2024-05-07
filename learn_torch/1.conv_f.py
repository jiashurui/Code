import torch.nn.functional as F
import torch

input = torch.tensor([[1, 2, 3, 4, 5],
                      [4, 5, 6, 7, 8],
                      [7, 8, 9, 10, 11],
                      [11, 12, 13, 14, 15],
                      [15, 16, 17, 18, 19],])
input = torch.reshape(input, (1,1,5,5))

print(input)
kernel = torch.tensor([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]])

kernel = torch.reshape(kernel, (1,1,3,3))
output = F.conv2d(input=input, weight=kernel, padding=1, stride=1, groups=1,dilation=1)

print(output)