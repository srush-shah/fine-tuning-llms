import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.split = split
        self.data = self.process_data(data_folder, split, self.tokenizer)
    
    def process_data(self, data_folder, split, tokenizer):
        # Load natural language and SQL files
        nl_path = os.path.join(data_folder, f'{split}.nl')
        nl_lines = load_lines(nl_path)
        
        processed_data = []
        
        if split == 'test':
            # For test set, we only have natural language queries
            for nl_query in nl_lines:
                processed_data.append({
                    'nl_query': nl_query,
                    'sql_query': None
                })
        else:
            # For train and dev, we have both NL and SQL
            sql_path = os.path.join(data_folder, f'{split}.sql')
            sql_lines = load_lines(sql_path)
            
            for nl_query, sql_query in zip(nl_lines, sql_lines):
                processed_data.append({
                    'nl_query': nl_query,
                    'sql_query': sql_query
                })
        
        return processed_data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        nl_query = item['nl_query']
        sql_query = item['sql_query']
        
        # Tokenize encoder input (natural language query)
        encoder_inputs = self.tokenizer(nl_query, return_tensors='pt', padding=False, truncation=True)
        encoder_ids = encoder_inputs['input_ids'].squeeze(0)
        
        if self.split == 'test':
            # For test set, return only encoder inputs
            return {
                'encoder_ids': encoder_ids,
                'sql_query': None
            }
        else:
            # For train/dev, tokenize decoder inputs and targets
            # In T5, decoder input includes BOS, and targets are what we predict (shifted)
            # Decoder input: add beginning of sentence token (extra_id_0) + SQL query
            decoder_text = f"<extra_id_0>{sql_query}"
            decoder_inputs = self.tokenizer(decoder_text, return_tensors='pt', padding=False, truncation=True)
            decoder_input_ids = decoder_inputs['input_ids'].squeeze(0)
            
            # Decoder targets: should be aligned with decoder input positions
            # decoder_input_ids: [BOS, token1, token2, ..., tokenN]
            # At position i, we predict decoder_input_ids[i+1]
            # So targets should be: [token1, token2, ..., tokenN, EOS] (same length as decoder_input)
            # This is decoder_input_ids[1:] + EOS
            eos_token_id = self.tokenizer.eos_token_id
            # Targets are decoder_input_ids[1:] + EOS to match the length
            target_ids = torch.cat([decoder_input_ids[1:], torch.tensor([eos_token_id])])
            
            # Initial decoder input is just the BOS token (for generation)
            initial_decoder_input = self.tokenizer("<extra_id_0>", return_tensors='pt', padding=False)['input_ids'].squeeze(0)
            
            return {
                'encoder_ids': encoder_ids,
                'decoder_input_ids': decoder_input_ids,
                'decoder_target_ids': target_ids,
                'initial_decoder_input': initial_decoder_input,
                'sql_query': sql_query
            }

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids_list = [item['encoder_ids'] for item in batch]
    decoder_input_ids_list = [item['decoder_input_ids'] for item in batch]
    decoder_target_ids_list = [item['decoder_target_ids'] for item in batch]
    initial_decoder_input_list = [item['initial_decoder_input'] for item in batch]
    
    # Pad sequences
    encoder_ids = pad_sequence(encoder_ids_list, batch_first=True, padding_value=PAD_IDX)
    decoder_inputs = pad_sequence(decoder_input_ids_list, batch_first=True, padding_value=PAD_IDX)
    
    # For targets, we need to align with decoder_inputs
    # decoder_inputs: [BOS, token1, token2, ..., tokenN] (length L)
    # targets should be: [token1, token2, ..., tokenN, EOS] (length L, aligned position-wise)
    # So we pad targets to match decoder_inputs length (which is already padded)
    max_decoder_len = decoder_inputs.size(1)  # Use the padded decoder_inputs length
    padded_targets = []
    for i, target_ids in enumerate(decoder_target_ids_list):
        # Each target should match its corresponding decoder_input length
        # But since decoder_inputs are already padded to max length, we pad targets to max length too
        target_len = len(target_ids)
        if target_len < max_decoder_len:
            padding = torch.full((max_decoder_len - target_len,), PAD_IDX, dtype=target_ids.dtype)
            padded_target = torch.cat([target_ids, padding])
        else:
            # Truncate if longer (shouldn't happen, but just in case)
            padded_target = target_ids[:max_decoder_len]
        padded_targets.append(padded_target)
    
    decoder_targets = torch.stack(padded_targets)
    
    # Create attention masks
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    # For initial decoder input, we just need the first token
    initial_decoder_inputs = torch.stack(initial_decoder_input_list)
    
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids_list = [item['encoder_ids'] for item in batch]
    
    # Pad sequences
    encoder_ids = pad_sequence(encoder_ids_list, batch_first=True, padding_value=PAD_IDX)
    
    # Create attention masks
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    # Get initial decoder input (BOS token)
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    bos_token_id = tokenizer.encode("<extra_id_0>", add_special_tokens=False)[0]
    batch_size = len(batch)
    initial_decoder_inputs = torch.full((batch_size, 1), bos_token_id, dtype=torch.long)
    
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(script_dir, 'data')
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x