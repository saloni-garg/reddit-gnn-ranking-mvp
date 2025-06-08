import pickle
import torch
from torch_geometric.data import HeteroData
import numpy as np
from torch.serialization import add_safe_globals
import networkx as nx
from src.common.edge_types import EDGE_TYPE_TO_ID


add_safe_globals(['torch_geometric.data.storage.BaseStorage'])


with open('src/data/heterogeneous_graph.gpickle', 'rb') as f:
    G = pickle.load(f)


data = HeteroData()



user_nodes = [n for n in G.nodes if 'type' in G.nodes[n] and G.nodes[n]['type'] == 'user']
post_nodes = [n for n in G.nodes if 'type' in G.nodes[n] and G.nodes[n]['type'] == 'post']
subreddit_nodes = [n for n in G.nodes if 'type' in G.nodes[n] and G.nodes[n]['type'] == 'subreddit']


user_id_map = {nid: i for i, nid in enumerate(user_nodes)}
post_id_map = {nid: i for i, nid in enumerate(post_nodes)}
subreddit_id_map = {nid: i for i, nid in enumerate(subreddit_nodes)}


user_feats = []
for n in user_nodes:
    node = G.nodes[n]
    user_feats.append([
        node.get('link_karma', 0),
        node.get('comment_karma', 0),
        node.get('activity', 0)
    ])
data['user'].x = torch.tensor(user_feats, dtype=torch.float)


post_feats = []
for n in post_nodes:
    node = G.nodes[n]
    emb = node.get('title_embedding', np.zeros(768))
    score = node.get('score', 0)
    num_comments = node.get('num_comments', 0)
    post_feats.append(np.concatenate([emb, [score, num_comments]]))
data['post'].x = torch.tensor(post_feats, dtype=torch.float)


subreddit_feats = []
for n in subreddit_nodes:
    node = G.nodes[n]
    cat = node.get('category_encoded', np.zeros(1))
    subscribers = node.get('subscribers', 0)
    over18 = int(node.get('over18', False))
    age = node.get('created_utc', 0)
    subreddit_feats.append(np.concatenate([cat, [subscribers, over18, age]]))
data['subreddit'].x = torch.tensor(subreddit_feats, dtype=torch.float)



for node_type in ['user', 'post', 'subreddit']:
    if hasattr(data[node_type], 'x'):
        data[node_type].num_nodes = data[node_type].x.shape[0]


for node_type in ['user', 'post', 'subreddit']:
    N = data[node_type].num_nodes
    edge_index = [[], []]
    edge_type = []
    for i in range(N):
        edge_index[0].append(i)
        edge_index[1].append(i)
        edge_type.append(EDGE_TYPE_TO_ID[(node_type, 'self', node_type)])
    data[(node_type, 'self', node_type)].edge_index = torch.tensor(edge_index, dtype=torch.long)
    data[(node_type, 'self', node_type)].edge_type = torch.tensor(edge_type, dtype=torch.long)


print("\nSelf-loop edge types after addition:")
for node_type in ['user', 'post', 'subreddit']:
    edge_type = (node_type, 'self', node_type)
    if edge_type in data.edge_types:
        print(f"{edge_type}: edge_index shape: {data[edge_type].edge_index.shape}, edge_type shape: {data[edge_type].edge_type.shape}")
    else:
        print(f"{edge_type}: Not found in data.edge_types")


edges = {
    'user': {
        'upvoted': {'post': {'0': [], '1': []}},
        'commented_on': {'post': {'0': [], '1': []}},
        'subscribed_to': {'subreddit': {'0': [], '1': []}},
    },
    'post': {
        'belongs_to': {'subreddit': {'0': [], '1': []}},
    },
}

for u, v, d in G.edges(data=True):
    if 'type' in d:
        if d['type'] == 'upvoted':
            edges['user']['upvoted']['post']['0'].append(u)
            edges['user']['upvoted']['post']['1'].append(v)
        elif d['type'] == 'commented_on':
            edges['user']['commented_on']['post']['0'].append(u)
            edges['user']['commented_on']['post']['1'].append(v)
        elif d['type'] == 'subscribed_to':
            edges['user']['subscribed_to']['subreddit']['0'].append(u)
            edges['user']['subscribed_to']['subreddit']['1'].append(v)
        elif d['type'] == 'belongs_to':
            edges['post']['belongs_to']['subreddit']['0'].append(u)
            edges['post']['belongs_to']['subreddit']['1'].append(v)


for src_type, rel, dst_type in sorted(EDGE_TYPE_TO_ID.keys(), key=lambda x: EDGE_TYPE_TO_ID[x]):
    if rel == 'self':  
        continue
    src_map = {'user': user_id_map, 'post': post_id_map, 'subreddit': subreddit_id_map}[src_type]
    dst_map = {'user': user_id_map, 'post': post_id_map, 'subreddit': subreddit_id_map}[dst_type]
    edge_index = [[], []]
    edge_type = []
    for u, v in zip(edges[src_type][rel][dst_type]['0'], edges[src_type][rel][dst_type]['1']):
        if u in src_map and v in dst_map:
            edge_index[0].append(src_map[u])
            edge_index[1].append(dst_map[v])
            edge_type.append(EDGE_TYPE_TO_ID[(src_type, rel, dst_type)])
    if edge_index[0]:  
        data[(src_type, rel, dst_type)].edge_index = torch.tensor(edge_index, dtype=torch.long)
        data[(src_type, rel, dst_type)].edge_type = torch.tensor(edge_type, dtype=torch.long)
    else:
        data[(src_type, rel, dst_type)].edge_index = torch.empty((2, 0), dtype=torch.long)
        data[(src_type, rel, dst_type)].edge_type = torch.empty((0,), dtype=torch.long)


for edge_type in data.edge_types:
    num_edges = data[edge_type].edge_index.shape[1]
    if num_edges > 0:
        data[edge_type].edge_type = torch.full(
            (num_edges,),
            fill_value=EDGE_TYPE_TO_ID[edge_type],
            dtype=torch.long,
        )


if ('post', 'belongs_to', 'subreddit') in data.edge_types:
    del data[('post', 'belongs_to', 'subreddit')]


torch.save(data, 'src/data/pyg_hetero_data.pt')
print('PyG HeteroData object saved to src/data/pyg_hetero_data.pt')

def load_hetero_data(path='src/data/pyg_hetero_data.pt'):
    """Load and return the PyG HeteroData object from file."""
    return torch.load(path, weights_only=False) 