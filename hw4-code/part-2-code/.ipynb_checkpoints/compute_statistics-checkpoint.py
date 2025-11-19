"""
Script to compute data statistics for Q4 of the assignment.
"""
import os
from collections import Counter
import re

def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def compute_statistics(data_folder, split):
    """Compute statistics for a given split."""
    nl_path = os.path.join(data_folder, f'{split}.nl')
    sql_path = os.path.join(data_folder, f'{split}.sql')
    
    nl_lines = load_lines(nl_path)
    sql_lines = load_lines(sql_path)
    
    # Number of examples
    num_examples = len(nl_lines)
    
    # Mean sentence length (in words)
    nl_lengths = [len(line.split()) for line in nl_lines]
    mean_nl_length = sum(nl_lengths) / len(nl_lengths) if nl_lengths else 0
    
    # Mean SQL query length (in tokens/words)
    sql_lengths = [len(line.split()) for line in sql_lines]
    mean_sql_length = sum(sql_lengths) / len(sql_lengths) if sql_lengths else 0
    
    # Vocabulary size (natural language) - unique words
    nl_vocab = set()
    for line in nl_lines:
        # Simple tokenization - split on whitespace and punctuation
        words = re.findall(r'\b\w+\b', line.lower())
        nl_vocab.update(words)
    nl_vocab_size = len(nl_vocab)
    
    # Vocabulary size (SQL) - unique tokens
    sql_vocab = set()
    for line in sql_lines:
        # SQL tokens - split on whitespace and common SQL separators
        tokens = re.findall(r'\b\w+\b', line.upper())
        sql_vocab.update(tokens)
    sql_vocab_size = len(sql_vocab)
    
    return {
        'num_examples': num_examples,
        'mean_nl_length': mean_nl_length,
        'mean_sql_length': mean_sql_length,
        'nl_vocab_size': nl_vocab_size,
        'sql_vocab_size': sql_vocab_size
    }

def compute_statistics_after_processing(data_folder, split):
    """Compute statistics after T5 tokenization."""
    from transformers import T5TokenizerFast
    
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    nl_path = os.path.join(data_folder, f'{split}.nl')
    sql_path = os.path.join(data_folder, f'{split}.sql')
    
    nl_lines = load_lines(nl_path)
    sql_lines = load_lines(sql_path)
    
    # Mean sentence length (in tokens after tokenization)
    nl_token_lengths = []
    for line in nl_lines:
        tokens = tokenizer.encode(line, add_special_tokens=False)
        nl_token_lengths.append(len(tokens))
    mean_nl_token_length = sum(nl_token_lengths) / len(nl_token_lengths) if nl_token_lengths else 0
    
    # Mean SQL query length (in tokens after tokenization)
    sql_token_lengths = []
    for line in sql_lines:
        tokens = tokenizer.encode(line, add_special_tokens=False)
        sql_token_lengths.append(len(tokens))
    mean_sql_token_length = sum(sql_token_lengths) / len(sql_token_lengths) if sql_token_lengths else 0
    
    # Vocabulary size (natural language) - unique token IDs
    nl_token_vocab = set()
    for line in nl_lines:
        tokens = tokenizer.encode(line, add_special_tokens=False)
        nl_token_vocab.update(tokens)
    nl_token_vocab_size = len(nl_token_vocab)
    
    # Vocabulary size (SQL) - unique token IDs
    sql_token_vocab = set()
    for line in sql_lines:
        tokens = tokenizer.encode(line, add_special_tokens=False)
        sql_token_vocab.update(tokens)
    sql_token_vocab_size = len(sql_token_vocab)
    
    return {
        'mean_nl_token_length': mean_nl_token_length,
        'mean_sql_token_length': mean_sql_token_length,
        'nl_token_vocab_size': nl_token_vocab_size,
        'sql_token_vocab_size': sql_token_vocab_size
    }

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(script_dir, 'data')
    
    print("=" * 60)
    print("Data Statistics BEFORE Pre-processing")
    print("=" * 60)
    
    for split in ['train', 'dev']:
        stats = compute_statistics(data_folder, split)
        print(f"\n{split.upper()} Set:")
        print(f"  Number of examples: {stats['num_examples']}")
        print(f"  Mean sentence length: {stats['mean_nl_length']:.2f}")
        print(f"  Mean SQL query length: {stats['mean_sql_length']:.2f}")
        print(f"  Vocabulary size (natural language): {stats['nl_vocab_size']}")
        print(f"  Vocabulary size (SQL): {stats['sql_vocab_size']}")
    
    print("\n" + "=" * 60)
    print("Data Statistics AFTER Pre-processing (T5 Tokenization)")
    print("=" * 60)
    
    for split in ['train', 'dev']:
        stats = compute_statistics_after_processing(data_folder, split)
        print(f"\n{split.upper()} Set:")
        print(f"  Mean sentence length (tokens): {stats['mean_nl_token_length']:.2f}")
        print(f"  Mean SQL query length (tokens): {stats['mean_sql_token_length']:.2f}")
        print(f"  Vocabulary size (natural language, token IDs): {stats['nl_token_vocab_size']}")
        print(f"  Vocabulary size (SQL, token IDs): {stats['sql_token_vocab_size']}")

