import os
import sys

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling
from torch_geometric.data import HeteroData
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from src.models.rgcn import HeteroRGCN
from src.data.pyg_graph_conversion import load_hetero_data

# Load data
data = torch.load('src/data/pyg_hetero_data.pt', weights_only=False)

# Link prediction: user upvoted post
edge_label_index = data['user', 'upvoted', 'post'].edge_index
num_edges = edge_label_index.size(1)

# Negative sampling
neg_edge_label_index = negative_sampling(
    edge_index=edge_label_index,
    num_nodes=(data['user'].num_nodes, data['post'].num_nodes),
    num_neg_samples=num_edges
)

# Prepare edge labels
edge_label = torch.cat([
    torch.ones(num_edges),
    torch.zeros(neg_edge_label_index.size(1))
])
edge_label_index = torch.cat([
    edge_label_index,
    neg_edge_label_index
], dim=1)

# Compute input feature dimensions for each node type
in_channels_dict = {ntype: data[ntype].x.size(1) for ntype in data.node_types}

# Ensure edge_type attributes are preserved in the batch
for edge_type in data.edge_types:
    if 'edge_type' not in data[edge_type]:
        num_edges = data[edge_type].edge_index.size(1)
        data[edge_type].edge_type = torch.zeros(num_edges, dtype=torch.long)
        print("-----djflksdjfkldsjlfdjlf-----")
        print(f"{edge_type} keys: {data[edge_type].keys()}")
    
class HeteroGNN(pl.LightningModule):
    def __init__(self, data, hidden_channels=128):
        super().__init__()
        self.data = data
        
        self.model = HeteroRGCN(
            metadata=data.metadata(),
            hidden_channels=hidden_channels
        )
        
        self.criterion = torch.nn.BCEWithLogitsLoss()
        
    def forward(self, x_dict, edge_index_dict, edge_type_dict=None):
        return self.model(x_dict, edge_index_dict, edge_type_dict)
    
    def training_step(self, batch, batch_idx):
        # Get node features and edge indices
        x_dict = {node_type: batch[node_type].x for node_type in self.data.node_types}
        edge_index_dict = {edge_type: batch[edge_type].edge_index for edge_type in self.data.edge_types}
        edge_type_dict = {edge_type: batch[edge_type].edge_type for edge_type in self.data.edge_types if hasattr(batch[edge_type], 'edge_type')}
        
        # Safeguard: skip batch if no users or posts
        num_users = batch['user'].x.size(0) if 'user' in batch and hasattr(batch['user'], 'x') else 0
        num_posts = batch['post'].x.size(0) if 'post' in batch and hasattr(batch['post'], 'x') else 0
        if num_users == 0 or num_posts == 0:
            return None  # or: return torch.tensor(0.0, requires_grad=True, device=self.device)
        
        # Forward pass
        out = self(x_dict, edge_index_dict, edge_type_dict)
        
        # Compute loss (example: link prediction between users and posts)
        pos_edge_index = batch[('user', 'upvoted', 'post')].edge_index
        neg_edge_index = self.negative_sampling(pos_edge_index, num_users, num_posts)
        if neg_edge_index is None:
            return None
        
        pos_pred = (out['user'][pos_edge_index[0]] * out['post'][pos_edge_index[1]]).sum(dim=1)
        neg_pred = (out['user'][neg_edge_index[0]] * out['post'][neg_edge_index[1]]).sum(dim=1)
        
        pred = torch.cat([pos_pred, neg_pred])
        target = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])
        
        loss = self.criterion(pred, target)
        self.log('train_loss', loss)
        return loss
    
    def negative_sampling(self, pos_edge_index, num_users, num_posts, num_neg_samples=None):
        if num_neg_samples is None:
            num_neg_samples = pos_edge_index.size(1)
        if num_users == 0 or num_posts == 0:
            return None
        # Generate random negative edges
        neg_edge_index = torch.randint(0, num_users, (2, num_neg_samples), device=pos_edge_index.device)
        neg_edge_index[1] = torch.randint(0, num_posts, (num_neg_samples,), device=pos_edge_index.device)
        return neg_edge_index
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

def main():
    # Load the heterogeneous graph
    data = load_hetero_data()
    
    # Print edge_type attributes for every edge type
    for edge_type in data.edge_types:
        if hasattr(data[edge_type], 'edge_type'):
            print(f"Edge type {edge_type} has edge_type attribute with shape {data[edge_type].edge_type.shape}")
        else:
            print(f"Edge type {edge_type} is missing edge_type attribute")
    
    # Create data loader
    loader = DataLoader(
        [data],  # Wrap data in a list since DataLoader expects a dataset
        batch_size=1,  # We can only use batch_size=1 since we're passing the whole graph
        shuffle=True
    )
    
    # Fetch and inspect the first batch from the DataLoader
    first_batch = next(iter(loader))
    print('--- Inspecting edge_type attributes in first DataLoader batch ---')
    for edge_type in first_batch.edge_types:
        if hasattr(first_batch[edge_type], 'edge_type'):
            print(f"Batch edge type {edge_type} has edge_type attribute with shape {first_batch[edge_type].edge_type.shape}")
        else:
            print(f"Batch edge type {edge_type} is missing edge_type attribute")
    print('---------------------------------------------------------------')
    
    # Create model
    model = HeteroGNN(data)
    
    # Train model
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, loader)

if __name__ == '__main__':
    main()

# Add edge_label and edge_label_index to data for PyG Lightning
# (PyG Lightning expects these fields for link prediction)
data['user', 'upvoted', 'post'].edge_label = edge_label
ndata = edge_label_index.size(1)
data['user', 'upvoted', 'post'].edge_label_index = edge_label_index

# DataLoader
loader = DataLoader(
    [data],  # Wrap data in a list since DataLoader expects a dataset
    batch_size=1,  # We can only use batch_size=1 since we're passing the whole graph
    shuffle=True
)

# Training
model = HeteroGNN(data)
trainer = pl.Trainer(max_epochs=5, accelerator='auto')
trainer.fit(model, loader)

# Save model
torch.save(model.state_dict(), 'src/models/rgcn_linkpred.pt')
print('Training complete! Model saved to src/models/rgcn_linkpred.pt') 