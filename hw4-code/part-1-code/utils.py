import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
import nltk
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation
    
    text = example["text"]
    words = word_tokenize(text)
    detokenizer = TreebankWordDetokenizer()
    
    # QWERTY keyboard layout for typo simulation
    qwerty_neighbors = {
        'a': ['s', 'q', 'w', 'z'],
        'e': ['w', 'r', 'd', 's'],
        'i': ['u', 'o', 'k', 'j'],
        'o': ['i', 'p', 'l', 'k'],
        'u': ['y', 'i', 'j', 'h'],
        's': ['a', 'd', 'w', 'e', 'x', 'z'],
        'd': ['s', 'f', 'e', 'r', 'c', 'x'],
        'f': ['d', 'g', 'r', 't', 'v', 'c'],
        'g': ['f', 'h', 't', 'y', 'b', 'v'],
        'h': ['g', 'j', 'y', 'u', 'n', 'b'],
        'j': ['h', 'k', 'u', 'i', 'm', 'n'],
        'k': ['j', 'l', 'i', 'o', 'm'],
        'l': ['k', 'o', 'p']
    }
    
    transformed_words = []
    for word in words:
        # Skip punctuation
        if not word.isalnum():
            transformed_words.append(word)
            continue
            
        word_lower = word.lower()
        is_upper = word[0].isupper() if len(word) > 0 else False
        
        # Apply synonym replacement with 20% probability
        if random.random() < 0.2:
            synsets = wordnet.synsets(word_lower)
            if synsets:
                # Get synonyms from the first synset
                synonyms = []
                for syn in synsets:
                    for lemma in syn.lemmas():
                        if lemma.name().lower() != word_lower and '_' not in lemma.name():
                            synonyms.append(lemma.name().lower())
                
                if synonyms:
                    synonym = random.choice(synonyms)
                    if is_upper:
                        synonym = synonym.capitalize()
                    transformed_words.append(synonym)
                    continue
        
        # Apply typos with 15% probability (only if synonym replacement didn't happen)
        if random.random() < 0.15:
            word_chars = list(word_lower)
            # Try to replace a vowel or common letter with a neighbor
            for i, char in enumerate(word_chars):
                if char in qwerty_neighbors and random.random() < 0.3:
                    neighbors = qwerty_neighbors[char]
                    if neighbors:
                        word_chars[i] = random.choice(neighbors)
                        break
            
            transformed_word = ''.join(word_chars)
            if is_upper:
                transformed_word = transformed_word.capitalize()
            transformed_words.append(transformed_word)
        else:
            transformed_words.append(word)
    
    example["text"] = detokenizer.detokenize(transformed_words)
    
    ##### YOUR CODE ENDS HERE ######

    return example
