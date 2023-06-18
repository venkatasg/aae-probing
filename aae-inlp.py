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
import ipdb

def run_inlp(num_classifiers, dialect, reps, seed):
    '''
    Main function that calls into inlp methods
    '''

    # hidden layer size
    hidden_size = reps.shape[1]

    # Split data
    x_train, x_dev, y_train, y_dev = train_test_split(reps, dialect, test_size=0.2, random_state=seed, shuffle=True)
    input_dim = x_train.shape[1]

    # Define classifier here
    clf = LinearSVC
    params = {"max_iter": 10000, "dual": False, "random_state": seed}


    _, _, Ws_rand, accs_rand = debias.get_random_projection(classifier_class=clf, cls_params=params, num_classifiers=num_classifiers, input_dim=hidden_size, is_autoregressive=True, min_accuracy=0, X_dev=x_dev, Y_dev=y_dev, by_class = False)

    _, _, Ws, accs = debias.get_debiasing_projection(classifier_class=clf, cls_params=params, num_classifiers=num_classifiers, input_dim=hidden_size, is_autoregressive=True, min_accuracy=0, X_train=x_train, Y_train=y_train, X_dev=x_dev, Y_dev=y_dev, by_class = False)

    print(accs, accs_rand)

    return Ws, Ws_rand

def run_rlace(num_classifiers, dialect, reps, seed):
    '''
    Main function that calls into inlp methods
    '''

    # hidden layer size
    hidden_size = reps.shape[1]

    # Split data
    x_train, x_dev, y_train, y_dev = train_test_split(reps, dialect, test_size=0.2, random_state=seed, shuffle=True)
    input_dim = x_train.shape[1]

    # Define classifier here
    num_iters = 50000
    rank=1
    optimizer_class = torch.optim.SGD
    optimizer_params_P = {"lr": 0.003, "weight_decay": 1e-4}
    optimizer_params_predictor = {"lr": 0.003,"weight_decay": 1e-4}
    epsilon = 0.001
    batch_size = 64


    _, _, Ws_rand, accs_rand = rlace.solve_adv_game(x_train, y_train, x_dev, y_dev, rank=rank, device="cpu", out_iters=num_iters, optimizer_class=optimizer_class, optimizer_params_P =optimizer_params_P, optimizer_params_predictor=optimizer_params_predictor, epsilon=epsilon,batch_size=batch_size)

    _, _, Ws, accs = rlace.solve_rand_game(x_train, y_train, x_dev, y_dev, rank=rank, device="cpu", out_iters=num_iters, optimizer_class=optimizer_class, optimizer_params_P =optimizer_params_P, optimizer_params_predictor=optimizer_params_predictor, epsilon=epsilon,batch_size=batch_size)

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
        default=48,
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
    reps = np.load('reps_diff/acts_layer_' + str(args.layer) + '_seed_' + str(args.seed) + '.npy')
    dialect = np.load('reps_diff/dialect_seed_' + str(args.seed) + '.npy')

    # Shuffling the arrays
    reps, dialect = shuffle(reps, dialect)

    Ws, Ws_rand = run_inlp(num_classifiers=args.num_classifiers, dialect=dialect, reps=reps, seed=args.seed)

    Ws = np.concatenate(Ws)
    Ws_rand = np.concatenate(Ws_rand)

    with open('reps_diff/Ws.layer={}.seed={}.npy'.format(args.layer, args.seed), 'wb') as f:
        np.save(f, Ws)

    with open('reps_diff/Ws.rand.layer={}.seed={}.npy'.format(args.layer, args.seed), 'wb') as f:
        np.save(f, Ws_rand)


