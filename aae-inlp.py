'''
Train iNLP classifier on tokens
'''

import numpy as np
import random
import ipdb
import argparse
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
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
    
    print("Classifier accuracies:", accs)
    print("Random accuracies:", accs_rand)    
    
    return Ws, Ws_rand


if __name__ == '__main__':
    # initialize argument parser
    description = 'Which layer to run inlp on'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--layer', type=int, default=6, help='Layer on which to run the SVM classifier')
    parser.add_argument('--num_classifiers', type=int, default=32, help='Number of inlp directions')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--cls', action='store_true', help='Train on CLS embeddings rather than word tokens')
    args = parser.parse_args()
    
    # Free up memory
    gc.collect()
    
    # Set random seeds for reproducibility on a specific machine
    random.seed(args.seed)
    np.random.seed(args.seed)
    np.random.RandomState(args.seed)
    
    # Load the representations
    if not args.cls:
        reps = np.load('reps_hate/hate_layer_' + str(args.layer) + '_seed_' + str(args.seed) + '.npy')   
        dialect = np.load('reps_hate/dialect_' + '_seed_' + str(args.seed) + '.npy') 
    else:    
        reps = np.load('reps_hate/cls_hate_layer_' + str(args.layer) + '_seed_' + str(args.seed) + '.npy')   
        dialect = np.load('reps_hate/cls_dialect_' + '_seed_' + str(args.seed) + '.npy')
    
    # Shuffling the arrays
    reps, dialect = shuffle(reps, dialect, random_state=args.seed)

    Ws, Ws_rand = run_inlp(num_classifiers=args.num_classifiers, dialect=dialect, reps=reps, seed=args.seed)
    
    Ws = np.concatenate(Ws)    
    Ws_rand = np.concatenate(Ws_rand)          
    
    with open("reps_hate/Ws.layer={}.iters={}.seed={}.cls={}.npy".format(args.layer, args.num_classifiers, args.seed, args.cls), "wb") as f:
        np.save(f, Ws)
        
    with open("reps_hate/P.layer={}.iters={}.seed={}.cls={}.npy".format(args.layer, args.num_classifiers, args.seed, args.cls), 'wb') as f:    
        np.save(f, P)
        
    
    