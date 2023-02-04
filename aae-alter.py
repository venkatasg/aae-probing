'''
Script to perform positive, negative and amnesic interventions on LM Representations
Positive intervention - make text more AAE
Negative intervention - make text more SAE
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
    set_seed,
    logging as loggingt
)
from accelerate import Accelerator
from datasets import Dataset, logging as loggingd
from inlp.debias import debias_by_specific_directions
from collections import Counter
from math import ceil

# Logging level for datasets
loggingd.disable_progress_bar()

 # Set logging to show only errors for transformers
loggingt.set_verbosity_error()

# Turn off annoying pandas warnings
pd.set_option('chained_assignment',None)

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)
    
def total_intervention(h_out, P, ws, alpha):
    '''
    Perform amnesic, positive or negative intervention
    alpha=0 : Amnesic
    alpha>0 : Positive
    alpha<0 : Negative
    '''

    h = h_out[0][:,1:,:]
    
    # Positive intervention is making text more AAVE
    signs = torch.sign(h@ws.T).long()
    
    # h_r component
    proj = (h@ws.T)
    if alpha>=0:
        proj = proj * signs
    else:
        proj = proj * (-signs)
    h_r = proj@ws*np.abs(alpha)
    
    # Get vector only in the direction of perpendicular to decision boundary
    h_n = h@P

    # Now pushing it either in positive or negative intervention direction
    h_alter = h_n + h_r
    
    # Return h_alter concatenated with the cls token
    return (torch.cat((h_out[0][:,:1,:], h_alter), dim=1),)


if __name__ == '__main__':
    # initialize argument parser
    description = 'Main intervention script'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--layer',
        type=int,
        help='Layer on which to run the SVM classifier',
        required=True
    )
    parser.add_argument(
        '--num_classifiers',
        type=int,
        required=True,
        help='Number of inlp directions to use'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='Random seed'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        required=True,
        help='Alpha for alterrep. Set to positive for positive intervention, negative for negative intervention, and zero for amnesic'
    )
    parser.add_argument(
        '--ceiling',
        type=float,
        default=0.3,
        help='\% of tokens to intervene on. 1 performs intervention on all tokens. Greater than 0 and less than 1.'
    )
    parser.add_argument(
        '--control',
        action='store_true',
        help='Use random subspace projections instead of inlp'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="cardiffnlp/twitter-roberta-base-offensive or tomh/toxigen-roberta or gronlp/hatebert",
        default='tomh/toxigen-roberta'
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to dataset",
        required="true"
    )
    parser.add_argument(
        "--reps_path",
        type=str,
        help="Path to reps",
        required="true"
    )
    parser.add_argument(
        '--write',
        action='store_true',
        help='Write predictions to file for analysis'
    )
    args = parser.parse_args()
    
    # Set random seeds for reproducibility on a specific machine
    set_seed(args.seed)
    random.seed(args.seed)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    device = accelerator.device
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)
    model.eval()
    
    # Load data
    df = pd.read_csv(args.data_path, sep='\t', quoting=csv.QUOTE_NONE)
    df['tweet'] = df['tweet'].apply(lambda x: preprocess(x))
    data = Dataset.from_pandas(df)
    data = data.map(lambda x: tokenizer(x['tweet'], truncation=True, padding="max_length", max_length=128), batched=True)
    
    data.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    # Extract attention mask to store and perform sampled intervention properly
    att_mask = data['attention_mask']
    att_mask_batched = [att_mask[i:i+args.batch_size] for i in range(0,att_mask.shape[0], args.batch_size)]
    
    dataloader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=False)
    
    # Sampled tokens Intervention hook function
    def intervention(h_out, P, ws, alpha, ceiling):
        '''
        Perform positive or negative intervention on all tokens
        '''
        
        # Get attention mask from nonlocal variables att_mask and batch_ind
        att_mask = att_mask_batched[batch_ind]
        
        # Collect altered representations in this varaible
        h_final = None
        
        for batch_elem in range(h_out[0].shape[0]):
            # Sample ceiling % of active tokens to perform intervention on
            h_elem = h_out[0][batch_elem]
            max_tok = torch.nonzero(att_mask[batch_elem]==1, as_tuple=True)[0][-1].item()
            sample_size = ceil(ceiling*max_tok)
            inds_to_alter = torch.randint(low=1, high=max_tok, size=(sample_size,))
            h_alter = h_elem[inds_to_alter,:]
            
            # AlterRep code starts here
            signs = torch.sign(h_alter@ws.T).long()
            
            # h_r component
            proj = (h_alter@ws.T)
            if alpha>=0:
                proj = proj * signs
            else:
                proj = proj * (-signs)
            h_r = proj@ws*np.abs(alpha)
            
            # Get vector only in the direction of perpendicular to decision boundary
            h_n = h_alter@P
        
            # Now pushing it either in positive or negative intervention direction
            h_alter = h_n + h_r
            
            for i, ind in enumerate(inds_to_alter):
                h_elem[ind,:] = h_alter[i,:]
            
            if h_final is None:
                h_final = h_elem.unsqueeze(0)
            else:
                h_final = torch.cat((h_final, h_elem.unsqueeze(0)), axis=0)
        
        # Return h_final
        return (h_final,)
    
    if args.num_classifiers > 0:
        # Load iNLP parameters
        if not args.control:
            with open(args.reps_path + "/Ws.layer={}.seed={}.npy".format(args.layer, args.seed), "rb") as f:
                Ws = np.load(f)
        else:
            with open(args.reps_path + "/Ws.rand.layer={}.seed={}.npy".format(args.layer, args.seed), "rb") as f:
                Ws = np.load(f)
    
        # Reduce Ws to number of classifiers you want
        Ws = Ws[:args.num_classifiers,:]
        
        # Now derive P from Ws
        list_of_ws = [np.array([Ws[i, :]]) for i in range(Ws.shape[0])]
        P = debias_by_specific_directions(
            directions=list_of_ws,
            input_dim=Ws.shape[1]
        )
        
        Ws = torch.tensor(Ws/np.linalg.norm(Ws, keepdims = True, axis = 1)).to(torch.float32).squeeze().to(device)
        P = torch.tensor(P).to(torch.float32).to(device)
        
        # Insert newaxis for 1 classifier edge case
        if len(Ws.shape) == 1:
            Ws = Ws[np.newaxis,:]
        
        # Attach hook
        if args.ceiling<1:
            hook = model.roberta.encoder.layer[args.layer-1].register_forward_hook(lambda m, h_in, h_out: intervention(h_out=h_out, P=P, ws=Ws, alpha=args.alpha, ceiling=args.ceiling))
        else:
            hook = model.roberta.encoder.layer[args.layer-1].register_forward_hook(lambda m, h_in, h_out: total_intervention(h_out=h_out, P=P, ws=Ws, alpha=args.alpha))

    dataloader, model = accelerator.prepare(dataloader, model)
    
    test_preds = np.array([])
    
    model.zero_grad()
    model.eval()
    
    for batch_ind, input_dict in enumerate(dataloader):
        with torch.no_grad():
            output = model(**input_dict)
            
        predictions = output['logits'].argmax(dim=1)
        test_preds = np.append(test_preds, predictions.detach().cpu().numpy().flatten())
    if args.num_classifiers > 0:
        hook.remove()
    
    df['pred'] = test_preds
    
    aave_dict = Counter(df[df['dialect']==1]['pred'].values)
    sae_dict = Counter(df[df['dialect']==0]['pred'].values)
    
    total_hate = df[df['pred']==1.0].shape[0]/df.shape[0]
    aave_hate = aave_dict[1.0]/(aave_dict[1.0] + aave_dict[0.0])
    sae_hate = sae_dict[1.0]/(sae_dict[1.0] + sae_dict[0.0])
    
    print("Hate speech % on all:", np.round(total_hate*100, 1))
    print("Hate speech % on AAE:", np.round(aave_hate*100, 1))
    print("Hate speech % on SAE:", np.round(sae_hate*100, 1))
    
    if args.write:
        df = df.loc[:, ['id', 'tweet', 'dialect', 'pred']]
        df.to_csv('dialect_post_intervention_seed_' + str(args.seed) + '_layer_' + str(args.layer) + '.tsv', sep='\t', quoting=csv.QUOTE_NONE, index=None)
