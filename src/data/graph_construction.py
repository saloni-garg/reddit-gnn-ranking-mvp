import pandas as pd
import networkx as nx
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from transformers import BertTokenizer, BertModel
import torch
import pickle

subreddits_df = pd.read_csv('src/data/subreddits.csv')
posts_df = pd.read_csv('src/data/posts.csv')
users_df = pd.read_csv('src/data/users.csv')

G = nx.DiGraph()

for _, row in subreddits_df.iterrows():
    G.add_node(row['id'], type='subreddit', name=row['name'], title=row['title'], description=row['description'], subscribers=row['subscribers'], created_utc=row['created_utc'], over18=row['over18'], category=row['category'])

for _, row in posts_df.iterrows():
    G.add_node(row['id'], type='post', title=row['title'], author=row['author'], subreddit=row['subreddit'], score=row['score'], num_comments=row['num_comments'], created_utc=row['created_utc'], selftext=row['selftext'])

for _, row in users_df.iterrows():
    G.add_node(row['name'], type='user', id=row['id'], link_karma=row['link_karma'], comment_karma=row['comment_karma'], created_utc=row['created_utc'])

for _, row in posts_df.iterrows():
    G.add_edge(row['author'], row['id'], type='upvoted')
    G.add_edge(row['id'], row['subreddit'], type='belongs_to')

for user in users_df['name'].sample(n=min(100, len(users_df))):
    for subreddit in subreddits_df['id'].sample(n=min(5, len(subreddits_df))):
        G.add_edge(user, subreddit, type='subscribed_to')

for user in users_df['name'].sample(n=min(100, len(users_df))):
    for post in posts_df['id'].sample(n=min(5, len(posts_df))):
        G.add_edge(user, post, type='commented_on')

categories = subreddits_df['category'].unique().reshape(-1, 1)
encoder = OneHotEncoder(sparse_output=False)
category_encoded = encoder.fit_transform(categories)
category_dict = {cat: enc for cat, enc in zip(categories.flatten(), category_encoded)}

for node in G.nodes():
    if 'type' in G.nodes[node] and G.nodes[node]['type'] == 'subreddit':
        cat = G.nodes[node]['category']
        G.nodes[node]['category_encoded'] = category_dict.get(cat, np.zeros(len(categories)))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

for node in G.nodes():
    if 'type' in G.nodes[node] and G.nodes[node]['type'] == 'post':
        title = G.nodes[node]['title']
        inputs = tokenizer(title, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        G.nodes[node]['title_embedding'] = outputs.last_hidden_state.mean(dim=1).numpy().flatten()

for node in G.nodes():
    if 'type' in G.nodes[node] and G.nodes[node]['type'] == 'user':
        G.nodes[node]['activity'] = G.nodes[node]['link_karma'] + G.nodes[node]['comment_karma']

with open('src/data/heterogeneous_graph.gpickle', 'wb') as f:
    pickle.dump(G, f)

print('Graph construction and feature engineering complete!') 