from textblob import TextBlob
from textstat.textstat import textstat
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import pandas as pd
import argparse
import string
import re


def extract_text_from_tweet(api_response):
    return api_response['text']


def remove_handles(content):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)"," ",content).split())


def count_handles(content):
    return len(re.findall("(@[A-Za-z0-9]+)",content))


def bool_handles(content):
    match = re.search("(@[A-Za-z0-9]+)", content)
    if match:
        return 1
    else: return 0


def count_hashtags(content):
    return len(re.findall("(#[A-Za-z0-9]+)",content))


def bool_hashtags(content):
    match = re.search("(#[A-Za-z0-9]+)", content)
    if match:
        return 1
    else: return 0


def is_retweet(content):
    return int("RT " in content)


def has_url(content):
    return int("https://" in content or "http://" in content)


def build_POS_list(content):
    content = content.decode('latin-1')
    return ' '.join([item[1] for item in pos_tag(word_tokenize(content))])


def feature_engineer_tweet(tweet):
    text = extract_text_from_tweet(tweet)
    
