# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 10:40:22 2018

@author: charliezou

This code is referenced from the NeaFM model code in the github url (https://github.com/hexiangnan/neural_collaborative_filtering)
"""


import numpy as np
import random

from keras import backend as K
from keras.models import Model
from keras.layers import Lambda, Embedding, Input, Add, Conv1D
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from Dataset import Dataset
from commonlayer import get_session_list, Position_Embedding
from evaluate import evaluate_scf
from time import time
import argparse

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run IPN.")
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
                        help='Number of attention layers.')   
    parser.add_argument('--gt', type=int, default=1,
                        help='Whether to global tracking.') 
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()
     
def get_model(num_users, seq_len, num_items, latent_dim, num_negatives, num_layer, gt=1):
    # Input variables
    u_input = Input(shape=(1,), dtype='int32', name = 'u_input')
    x_input = Input(shape=(seq_len,), dtype='int32', name = 'x_input')    
    session_input = Input(shape=(seq_len,), dtype='int32', name = 'session_input')
    
    item_input = Input(shape=(num_negatives+1,), dtype='int32', name = 'item_input')
    
    if gt > 0:
        T_Embedding_GT = Embedding(input_dim = num_users, output_dim = latent_dim, name = 't_gt_embedding')        
    K_Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_dim, name = 'k_item_embedding')
    K_Embedding_Session = Embedding(input_dim = seq_len, output_dim = latent_dim, name = 'session_embedding')
    T_Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_dim, name = 'item_embedding')     

    k_latent = K_Embedding_Item(x_input)  
    
    k_latent = Add()([k_latent, K_Embedding_Session(session_input)])
    #k_latent = Position_Embedding()(k_latent) 
    for i in range(num_layer):
        k_latent = Conv1D(latent_dim, 3, padding='same', activation='elu', name = 'conv_layer_{}'.format(i))(k_latent)
        
    user_latent = Lambda(lambda x: K.expand_dims(K.sum(x, axis=1), axis=1))(k_latent)
    if gt > 0:
        user_latent = Add()([user_latent, T_Embedding_GT(u_input)])
      
    item_latent = T_Embedding_Item(item_input)
        
    prediction = Lambda(lambda x: K.sigmoid(K.batch_dot(x[0][:,0,:], x[1],axes=(1,2))))([user_latent, item_latent])
    
    user_vec_model = Model(inputs=[u_input, x_input, session_input], outputs=user_latent)    
    model = Model(inputs=[u_input, x_input, session_input, item_input], outputs=prediction)
    
    model.summary()
            
    return model, user_vec_model
       
def get_train_instances(train, num_negatives, seq_len, num_items):
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

if __name__ == '__main__':
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
    model, u_model = get_model(num_users, seq_len, num_items, num_factors, num_negatives, num_layer, gt)
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
        u_input, x_input,session_input, item_input, labels = get_train_instances(train, num_negatives, seq_len, num_items)
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
                    model_out_file = 'Pretrain/%s_cscf_%d_%d_%d_%d_%d_%.4f_%.4f.h5' %( args.dataset, epoch, gt, seq_len, num_layer, num_negatives, best_hr, best_ndcg)
                    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best model is saved to %s" %(model_out_file))