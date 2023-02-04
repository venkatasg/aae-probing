'''
Collect model Representations for learning a classifier
'''

import torch
import pandas as pd
import random
import numpy as np
import ipdb
import csv
import argparse
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
    set_seed,
    logging as loggingt
)
from accelerate import Accelerator
from datasets import Dataset, logging as loggingd
import evaluate

# Logging level for datasets
loggingd.disable_progress_bar()

 # Set logging to show only errors for transformers
loggingt.set_verbosity_error()

# Turn off chained assignment warnings from pandas
pd.set_option('chained_assignment',None)

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

if __name__ == '__main__':
    # initialize argument parser
    description = 'Which layer to collect states on'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--layer',
        type=int,
        default=6,
        help='Layer on which to run inlp. Should be between 1 and 12'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='Random seed'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Some hate speech/toxic language classifier model",
        default='tomh/toxigen_roberta'
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to data",
        required=True
    )
    parser.add_argument(
        "--sampling_strat",
        type=str,
        help="Strategy for sampling representations. random, or diff",
        required=True
    )
    args = parser.parse_args()
    
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    device = accelerator.device
    
    # Set random seeds for reproducibility on a specific machine
    set_seed(args.seed)
    random.seed(args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, output_hidden_states=True)
    model.eval()
    
    # Build dataframe of data
    df = pd.read_csv(args.data_path, sep='\t', quoting=csv.QUOTE_NONE).sort_values('id').reset_index(drop=True)
    df['tweet'] = df['tweet'].apply(lambda x: preprocess(x))
    data = Dataset.from_pandas(df)
    data = data.map(lambda x: tokenizer(x['tweet'], truncation=True, padding="max_length", max_length=128), batched=True)
    # Rename group column to labels for use with BERT model
    data = data.rename_columns({'dialect': 'labels'})
    
    data.set_format(type='torch', columns=['id', 'input_ids', 'attention_mask', 'labels'])
    dataloader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=False)
    
    all_reps = np.array([], dtype=np.float32).reshape(0, 768)
    dialect = np.array([], dtype=np.float32).reshape(0, 1)
    
    # Prepare everything with our `accelerator`.
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for _, input_dict in enumerate(dataloader):
        with torch.no_grad():
            f_dict = {'input_ids': input_dict['input_ids'], 'attention_mask': input_dict['attention_mask']}
            outputs = model(**f_dict)
            
        # Extract ith layer activations
        layer_hidden_state = outputs['hidden_states'][args.layer].detach().cpu().numpy()
        
        # Sample tokens that are different in aave sentence from sae sentence?
        if args.sampling_strat=='diff':
            unique_ids =  input_dict['id'].unique()
            for _, unique_id in enumerate(unique_ids):
                # inds will be of length 2 for the dialect dataset we're using
                inds = torch.where(input_dict['id']==unique_id)[0]

                input_ids_1 = input_dict['input_ids'][inds[0], :]
                input_ids_2 = input_dict['input_ids'][inds[1], :]
                
                b_dialect_1 = input_dict['labels'][inds[0]].detach().cpu().numpy()
                b_dialect_2 = input_dict['labels'][inds[1]].detach().cpu().numpy()
                
                # Find token indices present in one but not the other
                toks_1 = [i for i, a in enumerate(input_ids_1) if a not in input_ids_2][:5]
                toks_2 = [i for i, a in enumerate(input_ids_2) if a not in input_ids_1][:5]
                
                # Get the sampled reps from the indices
                sampled_reps_1 = layer_hidden_state[inds[0], toks_1,:]
                sampled_reps_2 = layer_hidden_state[inds[1], toks_2,:]
                
                # Concatenate to the reps store
                all_reps = np.concatenate((all_reps, sampled_reps_1, sampled_reps_2), 0)
                
                # Get the dialect labels corresponding to each token rep
                rep_dialects = np.concatenate([np.repeat(b_dialect_1, len(toks_1), axis=0), np.repeat(b_dialect_2, len(toks_2), axis=0)])[:, np.newaxis]
                dialect = np.concatenate((dialect, rep_dialects), axis=0)
        else:
        # Sample 5 tokens at random from each sentence
            sampled_inds_dim1 = torch.tensor([], dtype=torch.int32)
            sampled_inds_dim0 = torch.tensor([], dtype=torch.int32)
            for i, sent in enumerate(input_dict['attention_mask']):
                # Length of sentence to set cut off for sampling
                max_tok = torch.nonzero(sent==1, as_tuple=True)[0][-1] - 1
                
                # Sample 5 words at random from each sentence
                s_inds = torch.randint(low=0, high=max_tok.detach().cpu().item(), size=(5,),  dtype=torch.long)
                
                sampled_inds_dim1 = torch.cat((sampled_inds_dim1, s_inds), axis=0)
                
                rep_dim0 =  torch.repeat_interleave(torch.tensor(i), 5)
                sampled_inds_dim0 = torch.cat((sampled_inds_dim0, rep_dim0), axis=0)
            
            sampled_reps = layer_hidden_state[sampled_inds_dim0, sampled_inds_dim1,:]
            all_reps = np.concatenate((all_reps, sampled_reps), 0)
            
            b_dialects = input_dict['labels'].detach().cpu().numpy()
            rep_dialects = np.repeat(b_dialects,[5 for i in range(b_dialects.shape[0])], axis=0)[:, np.newaxis]
            dialect = np.concatenate((dialect, rep_dialects), axis=0)
            
    # Save representations to file
    save_folder = 'reps_' + args.sampling_strat
    with open(save_folder + '/acts_layer_' + str(args.layer) + '_seed_' + str(args.seed) + '.npy', 'wb') as f:
        np.save(f, all_reps)
        
    with open(save_folder + '/dialect_seed_' + str(args.seed) + '.npy', 'wb') as f:
        np.save(f, dialect)
    
