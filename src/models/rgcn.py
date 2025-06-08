import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, RGCNConv
from collections import OrderedDict
from src.common.edge_types import EDGE_TYPES, EDGE_TYPE_TO_ID

class HeteroRGCN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels=128):
        super().__init__()
        # Explicit input dimensions for each node type
        in_channels = {
            'user': 3,  # [link_karma, comment_karma, activity]
            'post': 770,  # [BERT title embedding (768), score, num_comments]
            'subreddit': 9  # [category one-hot, subscribers, over18, age]
        }
        
        # Use shared edge types and mapping
        self.edge_types = EDGE_TYPES
        self.edge_type_to_id = EDGE_TYPE_TO_ID
        
        # Per-edge-type RGCNConv with correct number of relations and correct input dim
        self.conv1 = HeteroConv(
            OrderedDict([
                (et, RGCNConv(
                    hidden_channels,  # After root transform, all node features are hidden_channels
                    hidden_channels,
                    num_relations=len(self.edge_types),
                    root_weight=False,  # Disable root weight to avoid dimension mismatch
                    bias=True
                ))
                for et in self.edge_types
            ])
        )
        
        self.conv2 = HeteroConv(
            OrderedDict([
                (et, RGCNConv(
                    hidden_channels,
                    hidden_channels,
                    num_relations=len(self.edge_types),
                    root_weight=False,  # Disable root weight to avoid dimension mismatch
                    bias=True
                ))
                for et in self.edge_types
            ])
        )
        
        # Add separate linear layers for root transformations
        self.root_transform = nn.ModuleDict({
            node_type: nn.Linear(in_channels[node_type], hidden_channels)
            for node_type in in_channels.keys()
        })
        
        self.lin_user = nn.Linear(hidden_channels, hidden_channels)
        self.lin_post = nn.Linear(hidden_channels, hidden_channels)
        self.lin_subreddit = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_type_dict):
        # Map edge types to their corresponding relation IDs
        mapped_edge_type_dict = {}
        for edge_type, edge_type_tensor in edge_type_dict.items():
            if edge_type in self.edge_type_to_id:
                mapped_edge_type_dict[edge_type] = torch.full_like(
                    edge_type_tensor, 
                    self.edge_type_to_id[edge_type],
                    dtype=torch.long
                )
        
        # Only use edge types present in the batch
        non_empty_edge_index_dict = {k: v for k, v in edge_index_dict.items() if v.size(1) > 0}
        filtered_edge_type_dict = {k: v for k, v in mapped_edge_type_dict.items() if k in non_empty_edge_index_dict}
        
        # Apply root transformations first (project all input features to hidden_channels)
        for node_type, x in x_dict.items():
            if x.shape[1] != self.root_transform[node_type].in_features:
                raise ValueError(f"Input feature shape for {node_type} is {x.shape}, expected {self.root_transform[node_type].in_features}")
        x_dict = {
            node_type: self.root_transform[node_type](x)
            for node_type, x in x_dict.items()
        }
        for node_type, x in x_dict.items():
            if x.shape[1] != self.lin_user.in_features:
                raise ValueError(f"After root transform, {node_type} feature shape is {x.shape}, expected {self.lin_user.in_features}")
        
        # First convolution layer
        x_dict = self.conv1(x_dict, non_empty_edge_index_dict, filtered_edge_type_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        # Second convolution layer
        x_dict = self.conv2(x_dict, non_empty_edge_index_dict, filtered_edge_type_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        # Final linear layers
        return {
            'user': self.lin_user(x_dict['user']) if 'user' in x_dict else None,
            'post': self.lin_post(x_dict['post']) if 'post' in x_dict else None,
            'subreddit': self.lin_subreddit(x_dict['subreddit']) if 'subreddit' in x_dict else None
        } 