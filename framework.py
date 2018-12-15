# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 10:22:51 2018

@author: charliezou

Some of the code is referenced from https://github.com/hexiangnan/neural_collaborative_filtering 

"""

import numpy as np
import random

from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from Dataset import Dataset
from commonlayer import get_session_list
from evaluate import evaluate_scf
from time import time
import argparse


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run SCF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size.')    
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--seq_len', type=int, default=8,
                        help='Lenght of x input.')
    parser.add_argument('--num_layer', type=int, default=1,
                        help='Number of model layers.')   
    parser.add_argument('--gt', type=int, default=1,
                        help='Whether to global tracking for user.')     
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()


class SCF_Framework(object):
    """Abstract base SCF class.
    """
    
    def __init__(self, modelname, **kwargs):
        self.modelname = modelname
  
    def get_train_instances(self, train, num_negatives, seq_len, num_items):
        u_input, x_input, session_input, item_input, labels = [],[],[],[],[]
        
        for (u, r) in  train.items():
            h, ts, ng = r.items, r.ts, r.negItems        
            for i in range(len(h)-seq_len):
                x = h[i+1 : i+1+seq_len]
                s = get_session_list(ts[i+1 : i+1+seq_len])                 
                u_input.append([u])
                x_input.append(x)
                session_input.append(s)
                ii = [h[i]] + random.sample(ng, num_negatives)
                
                item_input.append(ii)
                labels.append([0])
        return u_input, x_input, session_input, item_input, labels
    
    def execution(self):
        args = parse_args()
        num_factors = args.num_factors
        num_negatives = args.num_neg
        num_layer= args.num_layer
        gt = args.gt
        seq_len = args.seq_len
        learner = args.learner
        learning_rate = args.lr
        epochs = args.epochs
        batch_size = args.batch_size
        verbose = args.verbose
        
        topK = 10
        print("arguments: %s" %(args))
        
        # Loading data
        t1 = time()
        dataset = Dataset(args.path + args.dataset)
        train, testRatings, testNegatives = dataset.trainRatings, dataset.testRatings, dataset.testNegatives
        num_users, num_items = dataset.num_users, dataset.num_items
        print("Load data done [%.1f s]. #user=%d, #item=%d" 
              %(time()-t1, num_users, num_items))
        
        # Build model
        model, u_model = self.get_model(num_users, seq_len, num_items, num_factors, num_negatives, num_layer, gt)
        if learner.lower() == "adagrad": 
            model.compile(optimizer=Adagrad(lr=learning_rate), loss='sparse_categorical_crossentropy')
        elif learner.lower() == "rmsprop":
            model.compile(optimizer=RMSprop(lr=learning_rate), loss='sparse_categorical_crossentropy')
        elif learner.lower() == "adam":
            model.compile(optimizer=Adam(lr=learning_rate), loss='sparse_categorical_crossentropy')
        else:
            model.compile(optimizer=SGD(lr=learning_rate), loss='sparse_categorical_crossentropy')    
        
        # Init performance
        t1 = time()
        (hits, ndcgs) = evaluate_scf(model, u_model, train, testRatings, testNegatives, topK, seq_len, num_items)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        print('Init: HR = %.4f, NDCG = %.4f\t [%.1f s]' % (hr, ndcg, time()-t1))
        
        # Train model
        best_hr, best_ndcg, best_iter = hr, ndcg, -1
        for epoch in range(1, 1+epochs):
            t0 = time()
            # Generate training instances
            u_input, x_input,session_input, item_input, labels = self.get_train_instances(train, num_negatives, seq_len, num_items)
            t1 = time()
            # Training
            hist = model.fit([np.array(u_input), np.array(x_input),np.array(session_input), np.array(item_input)], #input
                             np.array(labels), # labels 
                             batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
            t2 = time()
            
            # Evaluation
            if epoch %verbose == 0:
                (hits, ndcgs) = evaluate_scf(model, u_model, train, testRatings, testNegatives, topK, seq_len, num_items)
                hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
                print('Iteration %d [%.1f s, %.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                      % (epoch,  t2-t1, t1-t0, hr, ndcg, loss, time()-t2))
                if (hr+ndcg) > (best_hr+best_ndcg):
                    best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                    if args.out > 0:
                        model_out_file = 'Pretrain/%s_%s_%d_%d_%d_%d_%d_%.4f_%.4f.h5' %( args.dataset, self.modelname, epoch, gt, seq_len, num_layer, num_negatives, best_hr, best_ndcg)
                        model.save_weights(model_out_file, overwrite=True)
    
        print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
        if args.out > 0:
            print("The best model is saved to %s" %(model_out_file))  
    
       
    def get_model(self, num_users, seq_len, num_items, latent_dim, num_negatives, num_layer, gt=1):
        pass