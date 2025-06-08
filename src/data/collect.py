import os
import praw
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm


load_dotenv()

REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT')
REDDIT_USERNAME = os.getenv('REDDIT_USERNAME')
REDDIT_PASSWORD = os.getenv('REDDIT_PASSWORD')


reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT,
    username=REDDIT_USERNAME,
    password=REDDIT_PASSWORD
)

def collect_subreddits(limit=500):
    subreddits = []
    for subreddit in tqdm(reddit.subreddits.popular(limit=limit), desc='Collecting subreddits'):
        subreddits.append({
            'id': subreddit.id,
            'name': subreddit.display_name,
            'title': subreddit.title,
            'description': subreddit.public_description,
            'subscribers': subreddit.subscribers,
            'created_utc': subreddit.created_utc,
            'over18': subreddit.over18,
            'category': subreddit.advertiser_category or 'None',
        })
    return pd.DataFrame(subreddits)

def collect_posts(subreddit_names, limit=10):
    posts = []
    for name in tqdm(subreddit_names, desc='Collecting posts'):
        subreddit = reddit.subreddit(name)
        for post in subreddit.hot(limit=limit):
            posts.append({
                'id': post.id,
                'title': post.title,
                'author': str(post.author),
                'subreddit': name,
                'score': post.score,
                'num_comments': post.num_comments,
                'created_utc': post.created_utc,
                'selftext': post.selftext,
            })
    return pd.DataFrame(posts)

def collect_users(post_authors, limit=1):
    users = []
    for author in tqdm(set(post_authors), desc='Collecting users'):
        if author == 'None' or author is None:
            continue
        try:
            user = reddit.redditor(author)
            users.append({
                'name': user.name,
                'id': user.id,
                'link_karma': user.link_karma,
                'comment_karma': user.comment_karma,
                'created_utc': user.created_utc,
            })
        except Exception as e:
            continue
    return pd.DataFrame(users)

def main():
    subreddits_df = collect_subreddits(limit=50)  
    subreddits_df.to_csv('src/data/subreddits.csv', index=False)
    posts_df = collect_posts(subreddits_df['name'].tolist(), limit=20)
    posts_df.to_csv('src/data/posts.csv', index=False)
    users_df = collect_users(posts_df['author'].tolist())
    users_df.to_csv('src/data/users.csv', index=False)
    print('Data collection complete!')

if __name__ == '__main__':
    main() 