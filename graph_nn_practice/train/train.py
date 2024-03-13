import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.loader import NeighborLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset

target_dataset = 'ogbn-arxiv'# This will download the ogbn-arxiv to the 'networks' folder
dataset = PygNodePropPredDataset(name=target_dataset, root='networks')

from models.sage import SAGE

'''
Data(num_nodes=169343, edge_index=[2, 1166243], x=[169343, 128], node_year=[169343, 1], y=[169343, 1])

number of nodes, the adjacency list, 
the feature vector for the network, the year for each node, 
and the target label
'''
data = dataset[0]

split_idx = dataset.get_idx_split() 
        
train_idx = split_idx['train']
valid_idx = split_idx['valid']
test_idx = split_idx['test']

train_loader = NeighborLoader(data, input_nodes=train_idx,
                              shuffle=True, num_workers=os.cpu_count() - 2,
                              batch_size=1024, num_neighbors=[30] * 2)

total_loader = NeighborLoader(data, input_nodes=None, num_neighbors=[-1],
                               batch_size=4096, shuffle=False,
                               num_workers=os.cpu_count() - 2)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SAGE(data.x.shape[1], 256, dataset.num_classes, n_layers=2)
model.to(device)
epochs = 100
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
scheduler = ReduceLROnPlateau(optimizer, 'max', patience=7)


def test(model, device):
    evaluator = Evaluator(name=target_dataset)
    model.eval()
    out, var = model.inference(total_loader, device)

    y_true = data.y.cpu()
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    val_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, val_acc, test_acc, torch.mean(torch.Tensor(var))


for epoch in range(1, epochs):
    model.train()

    pbar = tqdm(total=train_idx.size(0))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0

    for batch in train_loader:
        batch_size = batch.batch_size
        optimizer.zero_grad()

        out, _ = model(batch.x.to(device), batch.edge_index.to(device))
        out = out[:batch_size]

        batch_y = batch.y[:batch_size].to(device)
        batch_y = torch.reshape(batch_y, (-1,))

        loss = F.nll_loss(out, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(batch_y).sum())
        pbar.update(batch.batch_size)

    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / train_idx.size(0)

    train_acc, val_acc, test_acc, var = test(model, device)
    
    print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}, Var: {var:.4f}')