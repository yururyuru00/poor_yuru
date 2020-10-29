import torch

t1 = torch.tensor([1, 2, 2])
t2 = torch.tensor([1, 0, 2])
t3 = torch.tensor([1, -1, 2])


print(torch.mm(t1, t2))
