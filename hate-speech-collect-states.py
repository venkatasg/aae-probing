'''
Collect BERT Representations for learning SVM classifier on top of
'''

import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
import random
from scipy.special import softmax
from datasets import Dataset
from datasets import logging
import torch
import gc
import ipdb
import csv
import argparse
from tqdm import tqdm

logging.disable_progress_bar()

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
    parser.add_argument('--layer', type=int, default=6, help='Layer on which to run the SVM classifier. Should be between 0 and 11')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--batch', type=int, default=128, help='Batch size')
    args = parser.parse_args()
    
    # Free up memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # Set random seeds for reproducibility on a specific machine
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    np.random.RandomState(args.seed)
    
    # Choose gpu or cpu
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(device)
    
    
    task='hate'
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, output_hidden_states=True).to(device)
    model.eval()
    
    # Build dataframe of data
    df_aave = pd.read_csv('dialect_data/aave_samples.tsv', sep='\t', quoting=csv.QUOTE_NONE)
    df_aave['dialect'] = 1
    df_aave['id'] = df_aave.index
    
    df_sae = pd.read_csv('dialect_data/sae_samples.tsv', sep='\t', quoting=csv.QUOTE_NONE)
    df_sae['dialect'] = 0
    df_sae['id'] = df_sae.index
    
    df = pd.concat([df_aave, df_sae])
    df.reset_index(inplace=True, drop=True)
    
    df['p_tweet'] = df['tweet'].apply(lambda x: preprocess(x))
    data = Dataset.from_pandas(df)
    data = data.map(lambda x: tokenizer(x['p_tweet'], truncation=True, padding="max_length", max_length=128), batched=True)
    
    data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'dialect'])
    dataloader = torch.utils.data.DataLoader(data, batch_size=args.batch)
    
    all_reps = np.array([], dtype=np.float32).reshape(0, 768)
    cls_reps = np.array([], dtype=np.float32).reshape(0, 768)
    dialect = np.array([], dtype=np.float32).reshape(0, 1)
    cls_dialect = df['dialect'].values
    
    with torch.no_grad():
        for _, input_dict in enumerate(dataloader):
            b_input_ids = input_dict['input_ids'].to(device)
            b_input_mask = input_dict['attention_mask'].to(device)
            b_dialects = input_dict['dialect'].detach().cpu().numpy()
            batch_size = b_input_ids.shape[0]
            
            outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask)
            hidden_states = outputs['hidden_states']

            # Extract ith layer activations
            ith_hidden_state = hidden_states[args.layer].detach().cpu().numpy()

           
            # Sample 5 tokens at random from each sentence
            sampled_inds_dim1 = torch.tensor([], dtype=torch.int32)
            sampled_inds_dim0 = torch.tensor([], dtype=torch.int32)
            for i, sent in enumerate(b_input_mask):
                max_tok = torch.nonzero(sent==1, as_tuple=True)[0][-1] - 1
                sampled_inds = torch.randint(1, max_tok.detach().cpu().item(), size=(5,))
                sampled_inds_dim1 = torch.cat((sampled_inds_dim1, sampled_inds), axis=0)
                
                rep_dim0 =  torch.repeat_interleave(torch.tensor(i), 5)
                sampled_inds_dim0 = torch.cat((sampled_inds_dim0, rep_dim0), axis=0)
            
            sampled_reps = ith_hidden_state[sampled_inds_dim0,sampled_inds_dim1,:]
            all_reps = np.concatenate((all_reps, sampled_reps), 0)
            cls_reps = np.concatenate((cls_reps,ith_hidden_state[:,0,:]), 0)
            
            rep_dialects = np.repeat(b_dialects,[5 for i in range(batch_size)], axis=0)[:, np.newaxis]
            dialect = np.concatenate((dialect, rep_dialects), axis=0)    
            
    # Save representations to file    
    with open('reps/hate_layer_' + str(args.layer) + '_seed_' + str(args.seed) + '.npy', 'wb') as f:    
        np.save(f, all_reps)
    
    with open('reps/cls_hate_layer_' + str(args.layer) + '_seed_' + str(args.seed) + '.npy', 'wb') as f:    
        np.save(f, cls_reps)
        
    with open('reps/dialect_' + '_seed_' + str(args.seed) + '.npy', 'wb') as f:    
        np.save(f, dialect)
        
    with open('reps/cls_dialect_' + '_seed_' + str(args.seed) + '.npy', 'wb') as f:
        np.save(f, cls_dialect)
    
