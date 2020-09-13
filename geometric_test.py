import torch
import argparse
from torch_geometric.datasets import Planetoid
from utilities import ExtractSubstructureContextPair

parser = argparse.ArgumentParser(
    description='PyTorch implementation of pre-training of GNN')
parser.add_argument('--l1', type=int, default=1,
                    help='l1 (default: 1).')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root='/tmp/Cora', name='Cora')
print(dataset[0])

# model = Net(dataset).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

'''model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()    

model.eval()
_, pred = model(data).max(dim=1)
correct = float (pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / data.test_mask.sum().item()
print('Accuracy: {:.4f}'.format(acc))'''
