from torch_geometric.datasets import CitationFull
import torch.utils.data

class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            **kwargs)


dataset = CitationFull(root='./data', name='Cora')
print(dataset[0])
loader = DataLoader(dataset, batch_size=1, shuffle=True)

for step, batch in enumerate(loader):
    print(batch)