"""This module defines a configurable SSTDataset class."""

import pytreebank
import torch
from loguru import logger
from pytorch_transformers import BertTokenizer
from torch.utils.data import Dataset
import pandas as pd

logger.info("Loading the tokenizer")
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

logger.info("Loading SST")
sst = pytreebank.load_sst()


def rpad(array, n=70):
    """Right padding."""
    current_len = len(array)
    if current_len > n:
        return array[: n - 1]
    extra = n - current_len
    return array + ([0] * extra)


def get_binary_label(label):
    """Convert fine-grained label to binary label."""
    if label < 2: #negative
        return 0
    if label > 2: #positive
        return 1
    raise ValueError("Invalid label")


class SSTDataset(Dataset):
    """Configurable SST Dataset.
    
    Things we can configure:
        - split (train / val / test)
        - root / all nodes
        - binary / fine-grained
    """

    def __init__(self, split="train", root=True, binary=True):
        """Initializes the dataset with given configuration.

        Args:
            split: str
                Dataset split, one of [train, val, test]
            root: bool
                If true, only use root nodes. Else, use all nodes.
            binary: bool
                If true, use binary labels. Else, use fine-grained.
        """
        logger.info(f"Loading SST {split} set")
        self.sst = sst[split]

        logger.info("Tokenizing")
        if root and binary:
            self.data = [
                (
                    rpad(
                        tokenizer.encode("[CLS] " + tree.to_lines()[0] + " [SEP]"), n=66
                    ),
                    get_binary_label(tree.label),
                )
                for tree in self.sst
                if tree.label != 2
            ]
        elif root and not binary:
            self.data = [
                (
                    rpad(
                        tokenizer.encode("[CLS] " + tree.to_lines()[0] + " [SEP]"), n=66
                    ),
                    tree.label,
                )
                for tree in self.sst
            ]
        elif not root and not binary:
            self.data = [
                (rpad(tokenizer.encode("[CLS] " + line + " [SEP]"), n=66), label)
                for tree in self.sst
                for label, line in tree.to_labeled_lines()
            ]
        else:
            self.data = [
                (
                    rpad(tokenizer.encode("[CLS] " + line + " [SEP]"), n=66),
                    get_binary_label(label),
                )
                for tree in self.sst
                for label, line in tree.to_labeled_lines()
                if label != 2
            ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X, y = self.data[index]
        X = torch.tensor(X)
        return X, y


##my code
tweets = pd.read_csv('airline_tweets_test.csv') 
tweets_text = list(tweets['text'])
tweets_labels = list(tweets['airline_sentiment'])

tweets_new_labels = []
for text, label in zip(tweets_text, tweets_labels):
    if label == 'negative':
        new_label = 0
        tweets_new_labels.append([text, new_label])
    if label =='positive':
        new_label = 1
        tweets_new_labels.append([text, new_label])

##

## changed original code from above
class TweetDataset(Dataset):
    """
    """

    def __init__(self):
        """Initializes the dataset with given configuration.

        """
        logger.info(f"Loading Tweets set") 
        self.tweets = tweets_new_labels 

        logger.info("Tokenizing")
        
        self.data = [
            (
                 rpad(
                    tokenizer.encode("[CLS] " + tweet[0] + " [SEP]"), n=66
                ),
                tweet[1],
            )
            for tweet in self.tweets
            ]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X, y = self.data[index]
        X = torch.tensor(X)
        return X, y

##