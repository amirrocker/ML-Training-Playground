from __future__ import print_function
import torch
import torch.nn as nn

x = torch.empty(5, 3)
print(x)

x = torch.randn(5, 3)
print(x)

x = torch.zeros(5, 3)
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# construct directly from data
xx = torch.tensor([5.4, .3543], dtype=torch.float)
print(xx)

tx = xx.new_ones(5, 3, dtype=torch.int)
print(tx)

ttx = torch.randn_like(x, dtype=torch.double)
print(ttx)
print(ttx.size())