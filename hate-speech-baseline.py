'''
Script that run's hate speech finetuned model on dialect dataset before intervention
'''

import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
import random
from scipy.special import softmax
from datasets import Dataset
from datasets import logging
import torch
import gc
import csv
import ipdb
from collections import Counter
from tqdm import tqdm
import argparse
from sklearn.metrics import confusion_matrix

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
    description = 'random seed'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
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
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device)
    model.eval()
    
    df_aave = pd.read_csv('dialect_data/aave_samples.tsv', sep='\t', quoting=csv.QUOTE_NONE)
    df_aave['dialect'] = 1
    df_aave['id'] = df_aave.index
    
    df_sae = pd.read_csv('dialect_data/sae_samples.tsv', sep='\t', quoting=csv.QUOTE_NONE)
    df_sae['dialect'] = -1
    df_sae['id'] = df_sae.index

    df = pd.concat([df_aave, df_sae])
    df.reset_index(inplace=True, drop=True)
    
    df['p_tweet'] = df['tweet'].apply(lambda x: preprocess(x))
    data = Dataset.from_pandas(df)
    data = data.map(lambda x: tokenizer(x['p_tweet'], truncation=True, padding="max_length", max_length=128), batched=True)
    
    data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'dialect'])
    dataloader = torch.utils.data.DataLoader(data, batch_size=128)
    
    map_hate_labels = ['not-hate', 'hate']
    map_dialect_labels ={-1: 'SAE', 1: 'AAVE'}
    pred_labels = []
    dialect_labels = []

    with torch.no_grad():
        for _, input_dict in enumerate(dataloader):
            b_input_ids = input_dict['input_ids'].to(device)
            b_input_mask = input_dict['attention_mask'].to(device)
            b_dialect = input_dict['dialect']
    
            output = model(input_ids=b_input_ids, attention_mask=b_input_mask)
            
            scores = output[0].detach().cpu().numpy()
            scores = softmax(scores, axis=1)
            b_labels = np.argmax(scores, axis=1)
            pred_labels += [map_hate_labels[x] for x in b_labels]
            dialect_labels += [x.item() for x in b_dialect]

    df['preds'] = pred_labels
    aave_dict = Counter(df[df['dialect']==1]['preds'].values)
    sae_dict = Counter(df[df['dialect']==-1]['preds'].values)
    
    aave_hate = aave_dict['hate']/(aave_dict['hate'] + aave_dict['not-hate'])
    sae_hate = sae_dict['hate']/(sae_dict['hate'] + sae_dict['not-hate'])
    print("Hate speech % on AAVE:", np.round(aave_hate, 4))
    print("Hate speech % on SAE:", np.round(sae_hate, 4))
    
    df = df.loc[:, ['id', 'tweet', 'dialect', 'preds']]
    
    df.to_csv('dialect_data/dialect_with_preds.csv', sep='\t', quoting=csv.QUOTE_NONE, index=None)