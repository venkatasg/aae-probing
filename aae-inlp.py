'''
Train iNLP classifier on tokens
'''

import pandas as pd
import numpy as np
import random
import ipdb
import argparse
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from inlp import debias


def run_inlp(num_classifiers, dialect, reps, seed):
    '''
    Main function that calls into inlp methods
    '''
    
    # Split data
    x_train, x_dev, y_train, y_dev = train_test_split(reps, dialect, test_size=0.2, random_state=seed, shuffle=True)
    input_dim = x_train.shape[1]
        
    # Define classifier here
    clf = LinearSVC
    params = {"max_iter": 10000, "dual": False, "random_state": seed}
    
    _, _, Ws_rand, accs_rand = debias.get_random_projection(classifier_class=clf, cls_params=params, num_classifiers=num_classifiers, input_dim=768, is_autoregressive=True, min_accuracy=0, X_dev=x_dev, Y_dev=y_dev, by_class = False)
    
    _, _, Ws, accs = debias.get_debiasing_projection(classifier_class=clf, cls_params=params, num_classifiers=num_classifiers, input_dim=768, is_autoregressive=True, min_accuracy=0, X_train=x_train, Y_train=y_train, X_dev=x_dev, Y_dev=y_dev, by_class = False)
    
    print(accs, accs_rand)
    
    return Ws, Ws_rand


if __name__ == '__main__':
    # initialize argument parser
    description = 'Which layer to run inlp on'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--layer',
        type=int,
        required=True,
        help='Layer on which to run the SVM classifier'
    )
    parser.add_argument(
        '--num_classifiers',
        type=int,
        default=64,
        help='Number of inlp directions'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='Random seed'
    )
    args = parser.parse_args()
    
    # Set random seeds for reproducibility on a specific machine
    random.seed(args.seed)
    np.random.seed(args.seed)
    np.random.RandomState(args.seed)
    
    # Load the representations
    reps = np.load('reps/acts_layer_' + str(args.layer) + '_seed_' + str(args.seed) + '.npy')
    dialect = np.load('reps/dialect_seed_' + str(args.seed) + '.npy')
    
    # Shuffling the arrays
    reps, dialect = shuffle(reps, dialect)

    Ws, Ws_rand = run_inlp(num_classifiers=args.num_classifiers, dialect=dialect, reps=reps, seed=args.seed)
    
    Ws = np.concatenate(Ws)
    Ws_rand = np.concatenate(Ws_rand)
    
    with open("reps/Ws.layer={}.seed={}.npy".format(args.layer, args.seed), "wb") as f:
        np.save(f, Ws)
        
    with open("reps/Ws.rand.layer={}.seed={}.npy".format(args.layer, args.seed), "wb") as f:
        np.save(f, Ws_rand)
    
    
