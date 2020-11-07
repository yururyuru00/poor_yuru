import torch
import itertools

tensor1 = torch.randn(4)
tensor2 = torch.randn(4,4)
tensor3 = torch.randn(4)

tensor1_ = torch.matmul(tensor1, tensor2)
out1 = torch.matmul(tensor1_, tensor3)
out2 = torch.FloatTensor(1)

tensor = torch.FloatTensor([0])
for i,j in itertools.combinations(range(3), 2):
    ij = torch.FloatTensor([i+j])
    tensor = torch.cat([tensor, ij], axis=0)
print(tensor)