'''
Script to perform positive, negative and amnesic interventions on CLS Representations
Positive intervention - make text more AAE
Negative intervention - make text more SAE
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
import argparse
import csv
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from inlp import debias
from tqdm import tqdm
from collections import Counter
from inlp.debias import get_rowspace_projection

logging.disable_progress_bar()

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)
    
def intervention(h_out, P, ws, cls, alpha):
    '''
    Perform amnesic, positive or negative intervention
    alpha=0 : Amnesic
    alpha>0 : Positive
    alpha<0 : Negative
    '''

    if cls:
        h_alter = h_out[0][:,:1,:]
    else:
        h_alter = h_out[0][:,1:,:]
        
    signs = torch.sign(h_alter@ws.T).long()
    
    # h_r component
    proj = (h_alter@ws.T) 
    if alpha>=0:
        proj = proj * signs
    else:
        proj = proj * (-signs)
    h_r = (proj@ws)*np.abs(alpha)
    
    # Get vector only in the direction of perpendicular to decision boundary
    h_n = h_alter@P

    # Now pushing it either in positive or negative intervention direction
    h_alter = h_n + h_r
        
    if cls:
        h_final = torch.cat((h_alter, h_out[0][:,1:,:]), dim=1)
    else:  
        h_final = torch.cat((h_out[0][:,:1,:], h_alter), dim=1)
   
    return (h_final,)

if __name__ == '__main__':
    # initialize argument parser
    description = 'Main intervention script'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--layer', type=int, help='Layer on which to run the SVM classifier', default=6)
    parser.add_argument('--num_classifiers', type=int, default=32, help='Number of inlp directions')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--batch', type=int, default=128, help='Batch size')
    parser.add_argument('--alpha', type=float, required=True, help='Alpha for alterrep. Set to positive for positive intervention, negative for negative intervention, and zero for amnesic')
    parser.add_argument('--cls', action='store_true', help='Intervene on all tokens rather than CLS embeddings')
    parser.add_argument('--control', action='store_true', help='Use random subspace projections instead of inlp ')
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

    # Load model    
    task='hate'
    MODEL = "cardiffnlp/twitter-roberta-base-hate"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device)
    model.eval() 
    
    # Load data
    df = pd.read_csv('dialect_data/dialect_with_preds.tsv', sep='\t', quoting=csv.QUOTE_NONE)   
    df['p_tweet'] = df['tweet'].apply(lambda x: preprocess(x))
    data = Dataset.from_pandas(df)
    data = data.map(lambda x: tokenizer(x['p_tweet'], truncation=True, padding="max_length", max_length=128), batched=True)
    
    data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'dialect'])
    dataloader = torch.utils.data.DataLoader(data, batch_size=args.batch, shuffle=False)

    # Load iNLP parameters
    if not args.control:
        with open("reps_hate/Ws.layer={}.iters={}.seed={}.cls={}.npy".format(args.layer, args.num_classifiers, args.seed, args.cls), "rb") as f:
            Ws = np.load(f)
        with open("reps_hate/P.layer={}.iters={}.seed={}.cls={}.npy".format(args.layer, args.num_classifiers, args.seed, args.cls), 'rb') as f:    
            P = np.load(f)
    else:
        Ws = np.random.rand(args.num_classifiers, 768) - 0.5                
        # Given rowspace projections, this calculates projection to intersection of nullspaces using formula from Ben-Israel 2013
        I = np.eye(768)
        P = I - get_rowspace_projection(Ws)

    Ws = torch.tensor(Ws/np.linalg.norm(Ws, keepdims=True, axis=1)).to(torch.float32).to(device)
    P = torch.tensor(P).to(torch.float32).to(device)
        
    map_hate_labels = ['not-hate', 'hate']
    pred_labels = []

    hook = model.roberta.encoder.layer[args.layer].register_forward_hook(lambda m, h_in, h_out: intervention(h_out=h_out, P=P, ws=Ws, cls=args.cls, alpha=args.alpha))
        
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
    
    hook.remove()

    df['pred'] = pred_labels
    aave_dict = Counter(df[df['dialect']==1]['pred'].values)
    sae_dict = Counter(df[df['dialect']==0]['pred'].values)
    
    aave_hate = aave_dict['hate']/(aave_dict['hate'] + aave_dict['not-hate'])
    sae_hate = sae_dict['hate']/(sae_dict['hate'] + sae_dict['not-hate'])
    print("Hate speech % on AAVE:", np.round(aave_hate, 3))
    print("Hate speech % on SAE:", np.round(sae_hate, 3))
    
    # df = df.loc[:, ['id', 'tweet', 'dialect', 'pred']]
    # df.to_csv('dialect_post_intervention.tsv', sep='\t', quoting=csv.QUOTE_NONE, index=None)