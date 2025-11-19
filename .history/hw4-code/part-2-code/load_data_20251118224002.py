import os
import json

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import T5TokenizerFast

PAD_IDX = 0
TASK_PREFIX = "translate English to SQL: "
MAX_SOURCE_LEN = 512  # Increased to accommodate schema
MAX_TARGET_LEN = 256

# Schema-aware training: Load and format schema
def load_schema(schema_path):
    """Load database schema from JSON file."""
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    return schema

def format_schema_summary(schema):
    """Format schema into a concise, SQL-like summary string for inclusion in prompts."""
    if not schema or 'ents' not in schema:
        return ""
    
    # Focus on most commonly used tables in the dataset
    # Based on analysis of the SQL queries, these are the key tables
    key_tables = ['flight', 'airport_service', 'city', 'airport', 'days', 'date_day', 
                  'airline', 'aircraft', 'flight_stop', 'fare', 'state']
    
    # Format as SQL-like CREATE TABLE statements for better learning
    tables_info = []
    for table_name in key_tables:
        if table_name in schema['ents']:
            columns = schema['ents'][table_name]
            if isinstance(columns, dict):
                # Include all columns but format more compactly
                col_names = list(columns.keys())
                # Format: "flight: flight_id, from_airport, to_airport, ..."
                tables_info.append(f"{table_name}: {', '.join(col_names)}")
    
    # Add other important tables
    other_important = ['equipment_sequence', 'ground_service', 'food_service', 'restriction']
    for table_name in other_important:
        if table_name in schema['ents']:
            columns = schema['ents'][table_name]
            if isinstance(columns, dict):
                col_names = list(columns.keys())[:8]  # Limit columns
                tables_info.append(f"{table_name}: {', '.join(col_names)}")
    
    # Format as a more structured schema description
    schema_text = "Schema: " + " | ".join(tables_info)
    return schema_text

# Load schema once at module level
_script_dir = os.path.dirname(os.path.abspath(__file__))
_schema_path = os.path.join(_script_dir, 'data', 'flight_database.schema')
_schema = None
_schema_summary = ""

try:
    _schema = load_schema(_schema_path)
    _schema_summary = format_schema_summary(_schema)
    # Truncate if too long (keep it concise for tokenization)
    if len(_schema_summary) > 400:
        _schema_summary = _schema_summary[:400] + "..."
    print(f"Loaded schema summary ({len(_schema_summary)} chars): {_schema_summary[:100]}...")
except Exception as e:
    print(f"Warning: Could not load schema: {e}")
    _schema_summary = ""

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
        
        # Tokenize encoder input (natural language query with task prefix and schema)
        # Include schema information to help model learn correct table/column names
        if _schema_summary:
            source_text = f"{TASK_PREFIX}{_schema_summary} | {nl_query}"
        else:
            source_text = f"{TASK_PREFIX}{nl_query}"
        
        encoder_inputs = self.tokenizer(
            source_text,
            return_tensors='pt',
            padding=False,
            truncation=True,
            max_length=MAX_SOURCE_LEN,
        )
        encoder_ids = encoder_inputs['input_ids'].squeeze(0)
        
        if self.split == 'test':
            return {
                'encoder_ids': encoder_ids,
            }
        
        # Tokenize decoder targets (SQL query)
        # T5 expects labels to include EOS token but NOT decoder start token
        # The model will automatically prepend pad_token_id as decoder start during training
        target_inputs = self.tokenizer(
            sql_query,
            return_tensors='pt',
            padding=False,
            truncation=True,
            max_length=MAX_TARGET_LEN,
            add_special_tokens=True,  # This adds EOS token, which T5 expects in labels
        )
        target_ids = target_inputs['input_ids'].squeeze(0)
        
        return {
            'encoder_ids': encoder_ids,
            'labels': target_ids,
        }

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns:
        * encoder_ids: Tensor of shape BxT for the encoder inputs.
        * encoder_mask: Attention mask of shape BxT (1 for non-pad, 0 otherwise).
        * labels: Tensor of shape BxT' containing the tokenized SQL targets.
    '''
    encoder_ids_list = [item['encoder_ids'] for item in batch]
    label_ids_list = [item['labels'] for item in batch]
    
    encoder_ids = pad_sequence(encoder_ids_list, batch_first=True, padding_value=PAD_IDX)
    labels = pad_sequence(label_ids_list, batch_first=True, padding_value=PAD_IDX)
    
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    return encoder_ids, encoder_mask, labels

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input.
    '''
    encoder_ids_list = [item['encoder_ids'] for item in batch]
    
    encoder_ids = pad_sequence(encoder_ids_list, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    return encoder_ids, encoder_mask

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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(data_folder):
        data_folder = os.path.join(script_dir, data_folder)

    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))

    return train_x, train_y, dev_x, dev_y, test_x