'''
Collect model Representations for learning a classifier
'''

import torch
import pandas as pd
import random
import numpy as np
import ipdb
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
        '--batch',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="cardiffnlp/twitter-roberta-base-hate or Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two",
        default='cardiffnlp/twitter-roberta-base-hate'
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
    df_aave = pd.read_csv('../data/dialect_data/aave_samples.tsv', sep='\t', quoting=csv.QUOTE_NONE)
    df_aave['dialect'] = 1
    df_aave['id'] = df_aave.index
    
    df_sae = pd.read_csv('../data/dialect_data/sae_samples.tsv', sep='\t', quoting=csv.QUOTE_NONE)
    df_sae['dialect'] = 0
    df_sae['id'] = df_sae.index
    
    df = pd.concat([df_aave, df_sae])
    df.reset_index(inplace=True, drop=True)
    
    df['tweet'] = df['tweet'].apply(lambda x: preprocess(x))
    data = Dataset.from_pandas(df)
    data = data.map(lambda x: tokenizer(x['tweet'], truncation=True, padding="max_length", max_length=128), batched=True)
    # Rename group column to labels for use with BERT model
    data = data.rename_columns({'dialect': 'labels'})
    
    data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    dataloader = torch.utils.data.DataLoader(data, batch_size=args.batch)
    
    all_reps = np.array([], dtype=np.float32).reshape(0, 768)
    dialect = np.array([], dtype=np.float32).reshape(0, 1)
    
    # Prepare everything with our `accelerator`.
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for _, input_dict in enumerate(dataloader):
        with torch.no_grad():
            output = model(**input_dict)
            
        # Extract ith layer activations
        layer_hidden_state = outputs['hidden_states'][args.layer].detach().cpu().numpy()
        
        # Sample tokens that are different in aave sentence from sae sentence?
        # ipdb.set_trace()
        
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
        
        sampled_reps = layer_hidden_state[sampled_inds_dim0, sampled_inds_dim1,:].detach().cpu().numpy()
        all_reps = np.concatenate((all_reps, sampled_reps), 0)
        
        b_dialects = input_dict['labels'].detach().cpu().numpy()
        rep_dialects = np.repeat(b_dialects,[5 for i in range(b_dialects.shape[0])], axis=0)[:, np.newaxis]
        dialect = np.concatenate((dialect, rep_dialects), axis=0)
            
    # Save representations to file
    with open('reps/acts_layer_' + str(args.layer) + '_seed_' + str(args.seed) + '.npy', 'wb') as f:
        np.save(f, all_reps)
        
    with open('reps/dialect_' + '_seed_' + str(args.seed) + '.npy', 'wb') as f:
        np.save(f, dialect)
    
