EDGE_TYPES = [
    ('user', 'self', 'user'),
    ('post', 'self', 'post'),
    ('subreddit', 'self', 'subreddit'),
    ('user', 'upvoted', 'post'),
    ('user', 'commented_on', 'post'),
    ('user', 'subscribed_to', 'subreddit'),
]

EDGE_TYPE_TO_ID = {etype: i for i, etype in enumerate(EDGE_TYPES)} 