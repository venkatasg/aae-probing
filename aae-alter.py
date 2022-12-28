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
import argparse
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
    logging as loggingt
)
from accelerate import Accelerator
from datasets import Dataset, logging as loggingd
import evaluate
from inlp.debias import debias_by_specific_directions

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
    
def intervention(h_out, P, ws, cls, alpha):
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
        '--control',
        action='store_true',
        help='Use random subspace projections instead of inlp'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="cardiffnlp/twitter-roberta-base-hate or Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two",
        default='cardiffnlp/twitter-roberta-base-hate'
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to dataset",
        required="true"
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
    dataloader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=False)
    
    if args.num_classifiers > 0:
        # Load iNLP parameters
        if not args.control:
            with open("reps_hate/Ws.layer={}.seed={}.npy".format(args.layer, args.seed), "rb") as f:
                Ws = np.load(f)
        else:
            with open("reps_hate/Ws.rand.layer={}.seed={}.npy".format(args.layer, args.seed), "rb") as f:
                Ws = np.load(f)
    
        # Reduce Ws to number of classifiers you want
        Ws = Ws[:args.num_classifiers,:]
        
        # Now derive P from Ws
        P = debias_by_specific_directions(
            directions=Ws,
            input_dim=Ws.shape[1]
        )
        
        Ws = torch.tensor(Ws/np.linalg.norm(Ws, keepdims = True, axis = 1)).to(torch.float32).squeeze().to(device)
        P = torch.tensor(P).to(torch.float32).to(device)
        
        # Insert newaxis for 1 classifier edge case
        if len(Ws.shape) == 1:
            Ws = Ws[np.newaxis,:]
        
        # Attach hook
        hook = model.roberta.encoder.layer[args.layer-1].register_forward_hook(lambda m, h_in, h_out: intervention(h_out=h_out, P=P, ws=Ws, cls=args.cls, alpha=args.alpha))

    test_dataloader, model = accelerator.prepare(test_dataloader, model)
    
    test_preds = np.array([])
    
    model.zero_grad()
    model.eval()
    
    for _, input_dict in enumerate(dataloader):
        with torch.no_grad():
            output = model(**input_dict)
            
        predictions = output['logits'].argmax(dim=1)
        test_preds = np.append(test_preds, predictions.detach().cpu().numpy().flatten())
    
    hook.remove()

    print("F1 Score: ", np.round(f1_metric.compute(average='micro')['f1']*100, 1))
    
    df['pred_dialect'] = test_preds
    
    aave_dict = Counter(df[df['dialect']==1]['pred_dialect'].values)
    sae_dict = Counter(df[df['dialect']==0]['pred_dialect'].values)
    aave_hate = aave_dict['hate']/(aave_dict['hate'] + aave_dict['not-hate'])
    sae_hate = sae_dict['hate']/(sae_dict['hate'] + sae_dict['not-hate'])
    
    print("Hate speech % on AAE:", np.round(aave_hate, 3))
    print("Hate speech % on SAE:", np.round(sae_hate, 3))
    
    df = df.loc[:, ['id', 'tweet', 'dialect', 'pred_dialect']]
    df.to_csv('dialect_post_intervention_seed_' + str(args.seed) + '_layer_' + str(args.layer) + '.tsv', sep='\t', quoting=csv.QUOTE_NONE, na_filter=False, index=None)
