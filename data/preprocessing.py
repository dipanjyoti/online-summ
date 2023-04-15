from pathlib import Path
from preprocessor import clean
from transformers import AutoTokenizer, DebertaModel
import torch
import os
import numpy as np

def extract_tweets(filename):
    tweets = []
    with open(filename, 'r') as file:
        for line in file:
            fields = line.split("\t")
            tweet = fields[3].strip()
            tweets.append(tweet)
        return tweets

def tweet_preprocessing(filename):
    tweets=extract_tweets(filename)
    cleaned_tweets = []

    for tweet in tweets:
        cleaned_tweet = clean(tweet)
        cleaned_tweets.append(cleaned_tweet)
    return cleaned_tweets


def data_embedding(tweets, args):

    root = Path(args.dataset_path)
    data_file = Path(args.dataset_file)

    embd_loc= root / data_file/ "raw" / f'{data_file}_embedding.npy'

    if os.path.exists(embd_loc):
        sentence_embeddings = np.load(embd_loc)
        return sentence_embeddings
    else:
        print ("\n embedding... \n")

        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
        model = DebertaModel.from_pretrained("microsoft/deberta-base")

        sentence_embeddings = []
        for tweet in tweets:

            inputs = tokenizer(tweet, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state

            sentence_embedding = torch.mean(last_hidden_states, dim=1)
            sentence_embedding = torch.round(sentence_embedding * 10000) / 10000 # Round the tensor to 4 decimals
            sentence_embeddings.append(sentence_embedding[0].detach().numpy())

        sentence_embeddings = np.array(sentence_embeddings)
        np.save(embd_loc, sentence_embeddings)

        return sentence_embeddings

def build(args):

    root = Path(args.dataset_path)
    assert root.exists(), f'provided datset path {root} does not exist'
    data_file = Path(args.dataset_file)
    dataset_loc= root / data_file/ "raw" / f'{data_file}_RAW_TWEET.txt'

    clean_tweets=tweet_preprocessing(dataset_loc)

    data=data_embedding(clean_tweets, args)

    return data
