from sklearn.model_selection import train_test_split

def split_tweets_to_arrays(tweet_df):
    X = tweet_df['tweet_text'].values
    y = tweet_df['labels'].values
    return X, y
