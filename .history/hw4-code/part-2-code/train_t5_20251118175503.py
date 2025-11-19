import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from transformers import GenerationConfig, T5TokenizerFast
from load_data import load_t5_data, PAD_IDX, TASK_PREFIX
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=1,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=20,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=5,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)

    args = parser.parse_args()
    return args

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    epochs_since_improvement = 0

    # Get script directory for path resolution
    script_dir = os.path.dirname(os.path.abspath(__file__))

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join(script_dir, 'checkpoints', f'{model_type}_experiments', args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir
    file_prefix = 't5_ft_experiment' if args.finetune else 't5_ft_experiment_ec'
    gt_sql_path = os.path.join(script_dir, 'data', 'dev.sql')
    gt_record_path = os.path.join(script_dir, 'records', 'ground_truth_dev.pkl')
    model_sql_path = os.path.join(script_dir, 'results', f'{file_prefix}_dev.sql')
    model_record_path = os.path.join(script_dir, 'records', f'{file_prefix}_dev.pkl')
    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(args, model, dev_loader,
                                                                         gt_sql_path, model_sql_path,
                                                                         gt_record_path, model_record_path)
        print(f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
        print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

        if args.use_wandb:
            result_dict = {
                'train/loss' : tr_loss,
                'dev/loss' : eval_loss,
                'dev/record_f1' : record_f1,
                'dev/record_em' : record_em,
                'dev/sql_em' : sql_em,
                'dev/error_rate' : error_rate,
            }
            wandb.log(result_dict, step=epoch)

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        save_model(checkpoint_dir, model, best=False)
        if epochs_since_improvement == 0:
            save_model(checkpoint_dir, model, best=True)

        if epochs_since_improvement >= args.patience_epochs:
            break

def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0

    for encoder_input, encoder_mask, decoder_targets in tqdm(train_loader):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        labels = decoder_targets.clone()
        labels[labels == PAD_IDX] = -100
        
        outputs = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            labels=labels,
        )
        
        loss = outputs.loss
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None: 
            scheduler.step()

        with torch.no_grad():
            non_pad = (labels != -100).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad

    return total_loss / total_tokens
        
def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    You must implement the evaluation loop to be using during training. We recommend keeping track
    of the model loss on the SQL queries, the metrics compute_metrics returns (save_queries_and_records should be helpful)
    and the model's syntax error rate. 

    To compute non-loss metrics, you will need to perform generation with the model. Greedy decoding or beam search
    should both provide good results. If you find that this component of evaluation takes too long with your compute,
    we found the cross-entropy loss (in the evaluation set) to be well (albeit imperfectly) correlated with F1 performance.
    '''
    model.eval()
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    total_loss = 0
    total_tokens = 0
    generated_queries = []
    
    with torch.no_grad():
        for batch in tqdm(dev_loader):
            encoder_input, encoder_mask, decoder_targets = batch
            
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)
            
            # Compute loss
            labels = decoder_targets.clone()
            labels[labels == PAD_IDX] = -100
            
            outputs = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                labels=labels,
            )
            
            loss = outputs.loss
            non_pad = (labels != -100).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad
            
            # Generate SQL queries
            # T5 uses pad_token_id as the decoder start token (which is 0)
            # The model config should have decoder_start_token_id set correctly
            decoder_start_token_id = model.config.decoder_start_token_id if hasattr(model.config, 'decoder_start_token_id') else tokenizer.pad_token_id
            
            generated = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_new_tokens=256,  # Generate up to 256 new tokens
                num_beams=3,  # Greedy decoding
                early_stopping=False,  # Only used with beam search, but set to False to avoid warning
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                decoder_start_token_id=decoder_start_token_id,
                repetition_penalty=1.2,  # Penalize repetition to prevent loops
                do_sample=False,
            )
            
            # Decode generated queries
            # T5's generate() for encoder-decoder models returns ONLY the decoder output
            # (not the encoder input), so we decode the entire sequence
            for gen_seq in generated:
                # Skip special tokens during decoding (this handles pad, eos, etc.)
                decoded = tokenizer.decode(gen_seq, skip_special_tokens=True)
                decoded = decoded.strip()
                # Remove task prefix if model accidentally generated it
                if decoded.startswith(TASK_PREFIX):
                    decoded = decoded[len(TASK_PREFIX):].strip()
                # Ensure we have a non-empty query
                if not decoded:
                    decoded = "SELECT 1"
                generated_queries.append(decoded)
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0

    # Debug: Print first few generated queries to see what's being produced
    if len(generated_queries) > 0:
        print(f"\nDebug: First 3 generated queries:")
        for i, q in enumerate(generated_queries[:3]):
            print(f"  {i+1}: {q[:100]}...")
    
    # Save generated queries and compute metrics
    save_queries_and_records(generated_queries, model_sql_path, model_record_path)
    sql_em, record_em, record_f1, error_msgs = compute_metrics(gt_sql_pth, model_sql_path, gt_record_path, model_record_path)
    
    # Compute error rate
    error_count = sum(1 for msg in error_msgs if msg != "")
    error_rate = error_count / len(error_msgs) if len(error_msgs) > 0 else 0
    
    return avg_loss, record_f1, record_em, sql_em, error_rate
        
def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    You must implement inference to compute your model's generated SQL queries and its associated 
    database records. Implementation should be very similar to eval_epoch.
    '''
    model.eval()
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    generated_queries = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            encoder_input, encoder_mask = batch
            
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            
            # Generate SQL queries
            # T5 uses pad_token_id as the decoder start token (which is 0)
            # The model config should have decoder_start_token_id set correctly
            decoder_start_token_id = model.config.decoder_start_token_id if hasattr(model.config, 'decoder_start_token_id') else tokenizer.pad_token_id
            
            generated = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_new_tokens=256,  # Generate up to 256 new tokens
                num_beams=3,  # Greedy decoding
                early_stopping=False,  # Only used with beam search, but set to False to avoid warning
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                decoder_start_token_id=decoder_start_token_id,
                repetition_penalty=1.2,  # Penalize repetition
                do_sample=False,
            )
            
            # Decode generated queries
            # T5's generate() for encoder-decoder models returns ONLY the decoder output
            # (not the encoder input), so we decode the entire sequence
            for gen_seq in generated:
                # Skip special tokens during decoding (this handles pad, eos, etc.)
                decoded = tokenizer.decode(gen_seq, skip_special_tokens=True)
                decoded = decoded.strip()
                # Remove task prefix if model accidentally generated it
                if decoded.startswith(TASK_PREFIX):
                    decoded = decoded[len(TASK_PREFIX):].strip()
                # Ensure we have a non-empty query
                if not decoded:
                    decoded = "SELECT 1"
                generated_queries.append(decoded)
    
    # Save generated queries and records
    save_queries_and_records(generated_queries, model_sql_path, model_record_path)

def main():
    # Get key arguments
    args = get_args()
    if args.use_wandb:
        # Recommended: Using wandb (or tensorboard) for result logging can make experimentation easier
        setup_wandb(args)

    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # Train 
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Evaluate
    model = load_model_from_checkpoint(args, best=True)
    model.eval()

    # Get script directory for path resolution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Dev set
    experiment_name = 'ft_experiment'
    model_type = 'ft' if args.finetune else 'scr'
    file_prefix = 't5_ft_experiment' if args.finetune else 't5_ft_experiment_ec'
    gt_sql_path = os.path.join(script_dir, 'data', 'dev.sql')
    gt_record_path = os.path.join(script_dir, 'records', 'ground_truth_dev.pkl')
    model_sql_path = os.path.join(script_dir, 'results', f'{file_prefix}_dev.sql')
    model_record_path = os.path.join(script_dir, 'records', f'{file_prefix}_dev.pkl')
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(args, model, dev_loader,
                                                                                    gt_sql_path, model_sql_path,
                                                                                    gt_record_path, model_record_path)
    print(f"Dev set results: Loss: {dev_loss:.4f}, Record F1: {dev_record_f1:.4f}, Record EM: {dev_record_em:.4f}, SQL EM: {dev_sql_em:.4f}")
    print(f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")

    # Test set
    model_sql_path = os.path.join(script_dir, 'results', f'{file_prefix}_test.sql')
    model_record_path = os.path.join(script_dir, 'records', f'{file_prefix}_test.pkl')
    test_inference(args, model, test_loader, model_sql_path, model_record_path)

if __name__ == "__main__":
    main()
