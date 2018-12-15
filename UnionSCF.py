# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 10:40:22 2018

@author: charliezou

Some of the code is referenced from https://github.com/hexiangnan/neural_collaborative_filtering

"""


import numpy as np
import random


from keras import backend as K
from keras.models import Model
from keras.layers import Lambda, Embedding, Input, Add, Conv1D, Concatenate
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from Dataset import Dataset
from commonlayer import get_session_list, create_n_attnlayer, Position_Embedding
from evaluate import evaluate_scf
from time import time
import argparse

from AttentionSCF import ASCF
from ConvSCF import CSCF
from SimpleSCF import SSCF

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
    parser.add_argument('--num_factors', type=int, default=4,
                        help='Embedding size.')    
    parser.add_argument('--num_neg', type=int, default=8,
                        help='Number of negative instances to pair with a positive instance.')       
    parser.add_argument('--seq_len', nargs='?', default='[12,12,4]',
                        help="Lenghts of x input of ascf|cscf |sscf model")    
    parser.add_argument('--num_layer', nargs='?', default='[2,2,1]',
                        help="Numbers of layers of ascf|cscf |sscf model")    
    parser.add_argument('--ascf_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for ascf part. If empty, no pretrain model will be used')
    parser.add_argument('--cscf_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for cscf part. If empty, no pretrain model will be used')
    parser.add_argument('--sscf_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for sscf part. If empty, no pretrain model will be used')    
    parser.add_argument('--alpha', nargs='?', default='[0.5,0.5,0.5]',
                        help="Weight ratios of ascf|cscf |sscf model")       
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')

    return parser.parse_args()

#return [:, 1, latent_dim]
def get_ascf_vec(seq_len, num_items, latent_dim, num_layer, x_input, session_input):
    K_Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_dim, name = 'ascf_k_item_embedding')
    Q_Embedding_Item = Embedding(input_dim = 1, output_dim = latent_dim, name = 'ascf_q_item_embedding')
    K_Embedding_Session = Embedding(input_dim = seq_len, output_dim = latent_dim, name = 'ascf_session_embedding') 
                    
    k_latent = K_Embedding_Item(Lambda(lambda x: x[:, :seq_len])(x_input))          
    k_latent = Add()([k_latent, K_Embedding_Session(Lambda(lambda x: x[:, :seq_len])(session_input))])

    q_input =Lambda(lambda x: K.zeros((K.shape(x)[0], 1), dtype='int32'))(x_input)
    q_latent = Q_Embedding_Item(q_input)    
    
    vec, _ = create_n_attnlayer(latent_dim, num_items, num_layer, x_input, q_latent, k_latent)        
    return vec
#return [:, 1, latent_dim]
def get_cscf_vec(seq_len, num_items, latent_dim, num_layer, x_input, session_input):
    K_Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_dim, name = 'cscf_k_item_embedding')
    K_Embedding_Session = Embedding(input_dim = seq_len, output_dim = latent_dim, name = 'cscf_session_embedding')         
        
    k_latent = K_Embedding_Item(Lambda(lambda x: x[:, :seq_len])(x_input))      
    k_latent = Position_Embedding()(k_latent)    
    k_latent = Add()([k_latent, K_Embedding_Session(Lambda(lambda x: x[:, :seq_len])(session_input))])
    
    for i in range(num_layer):
        k_latent = Conv1D(latent_dim, 3, padding='same', activation='elu', name = 'conv_layer_{}'.format(i))(k_latent)
        
    vec = Lambda(lambda x: K.expand_dims(K.sum(x, axis=1), axis=1))(k_latent)        
    return vec

def get_sscf_vec(seq_len, num_items, latent_dim, num_layer, x_input, session_input):
    K_Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_dim, name = 'sscf_k_item_embedding')
    K_Embedding_Session = Embedding(input_dim = seq_len, output_dim = latent_dim, name = 'sscf_session_embedding') 
        
    k_latent = K_Embedding_Item(Lambda(lambda x: x[:, :seq_len])(x_input))      
    k_latent = Add()([k_latent, K_Embedding_Session(Lambda(lambda x: x[:, :seq_len])(session_input))])
            
    vec = Lambda(lambda x: K.expand_dims(K.sum(x, axis=1), axis=1))(k_latent)        
    return vec
     
def get_model(num_users, seq_len, num_items, latent_dim, num_negatives, num_layer):
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    x_input = Input(shape=(max(seq_len),), dtype='int32', name = 'x_input')
    session_input = Input(shape=(max(seq_len),), dtype='int32', name = 'session_input')    
    item_input = Input(shape=(num_negatives+1,), dtype='int32', name = 'item_input')        
    
    T_Embedding_GT = Embedding(input_dim = num_users, output_dim = 3* latent_dim, name = 't_gt_embedding')            
    T_Embedding_Item = Embedding(input_dim = num_items, output_dim = 3*latent_dim, name = 'item_embedding')
    
    seq2vec_ascf = get_ascf_vec(seq_len[0], num_items, latent_dim, num_layer[0], x_input, session_input)    
    seq2vec_cscf = get_cscf_vec(seq_len[1], num_items, latent_dim, num_layer[1], x_input, session_input)
    seq2vec_sscf = get_sscf_vec(seq_len[2], num_items, latent_dim, num_layer[2], x_input, session_input)
    
    user_latent = Concatenate()([seq2vec_ascf, seq2vec_cscf, seq2vec_sscf])          
    user_latent = Add()([user_latent, T_Embedding_GT(user_input)])
      
    item_latent = T_Embedding_Item(item_input)
        
    prediction = Lambda(lambda x: K.sigmoid(K.batch_dot(x[0][:,0,:], x[1],axes=(1,2))))([user_latent, item_latent])
    
    user_vec_model = Model(inputs=[user_input, x_input, session_input], outputs=user_latent)    
    model = Model(inputs=[user_input, x_input, session_input, item_input], outputs=prediction)
    
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

def load_pretrain_model(model, ascf_model, cscf_model, sscf_model, num_layer, alpha):
    # load ascf  
    ascf_t_gt_embedding = ascf_model.get_layer('t_gt_embedding').get_weights()
    ascf_k_item_embedding = ascf_model.get_layer('k_item_embedding').get_weights()
    ascf_q_item_embedding = ascf_model.get_layer('q_item_embedding').get_weights()
    ascf_session_embedding = ascf_model.get_layer('session_embedding').get_weights()
    ascf_item_embedding = ascf_model.get_layer('item_embedding').get_weights()
    
    model.get_layer('ascf_k_item_embedding').set_weights(ascf_k_item_embedding)
    model.get_layer('ascf_q_item_embedding').set_weights(ascf_q_item_embedding)
    model.get_layer('ascf_session_embedding').set_weights(ascf_session_embedding)
        
    for layer_id in range(num_layer[0]):
        attn_v_embedding = ascf_model.get_layer('v_embedding_layer_{}'.format(layer_id)).get_weights()
        model.get_layer('v_embedding_layer_{}'.format(layer_id)).set_weights(attn_v_embedding)
          
    # load cscf
    cscf_t_gt_embedding = cscf_model.get_layer('t_gt_embedding').get_weights()
    cscf_k_item_embedding = cscf_model.get_layer('k_item_embedding').get_weights()
    cscf_session_embedding = cscf_model.get_layer('session_embedding').get_weights()
    cscf_item_embedding = cscf_model.get_layer('item_embedding').get_weights()
    
    model.get_layer('cscf_k_item_embedding').set_weights(cscf_k_item_embedding)
    model.get_layer('cscf_session_embedding').set_weights(cscf_session_embedding)

    for i in range(num_layer[1]):
        conv_weight = cscf_model.get_layer('conv_layer_{}'.format(i)).get_weights()
        model.get_layer('conv_layer_{}'.format(i)).set_weights(conv_weight)
        
    # load sscf
    sscf_t_gt_embedding = sscf_model.get_layer('t_gt_embedding').get_weights()
    sscf_k_item_embedding = sscf_model.get_layer('k_item_embedding').get_weights()
    sscf_session_embedding = sscf_model.get_layer('session_embedding').get_weights()
    sscf_item_embedding = sscf_model.get_layer('item_embedding').get_weights()
    
    model.get_layer('sscf_k_item_embedding').set_weights(sscf_k_item_embedding)
    model.get_layer('sscf_session_embedding').set_weights(sscf_session_embedding)    
    
    # load other
    t_gt_embedding = np.concatenate((ascf_t_gt_embedding[0], cscf_t_gt_embedding[0], sscf_t_gt_embedding[0]), axis=1)
    model.get_layer('t_gt_embedding').set_weights([t_gt_embedding])
    
    item_embedding = np.concatenate((ascf_item_embedding[0]*alpha[0], cscf_item_embedding[0]*alpha[1], sscf_item_embedding[0]*alpha[2]), axis=1)
    model.get_layer('item_embedding').set_weights([item_embedding])
      
    return model

if __name__ == '__main__':
    args = parse_args()
    num_factors = args.num_factors
    num_negatives = args.num_neg
    
    num_layer = eval(args.num_layer)
    seq_len = eval(args.seq_len)
    
   
    learner = args.learner
    learning_rate = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    verbose = args.verbose
    
    ascf_pretrain = 'Pretrain/' + args.ascf_pretrain
    cscf_pretrain = 'Pretrain/' + args.cscf_pretrain
    sscf_pretrain = 'Pretrain/' + args.sscf_pretrain
    
    alpha = eval(args.alpha)
    
    topK = 10
    print("arguments: %s" %(args))
    model_out_file = 'Pretrain/%s_%d_%d.h5' %(args.dataset, num_factors, time())
    
    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainRatings, dataset.testRatings, dataset.testNegatives
    num_users, num_items = dataset.num_users, dataset.num_items
    print("Load data done [%.1f s]. #user=%d, #item=%d" 
          %(time()-t1, num_users, num_items))
    
    # Build model
    model, u_model = get_model(num_users, seq_len, num_items, num_factors, num_negatives, num_layer)
    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='sparse_categorical_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='sparse_categorical_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='sparse_categorical_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='sparse_categorical_crossentropy')    
    
    # Load pretrain model
    if args.ascf_pretrain != '' and args.cscf_pretrain != '' and args.sscf_pretrain != '':
        ascf_model,_ = ASCF().get_model(num_users, seq_len[0], num_items, num_factors, num_negatives, num_layer[0])
        ascf_model.load_weights(ascf_pretrain)
        cscf_model,_ = CSCF().get_model(num_users, seq_len[1], num_items, num_factors, num_negatives, num_layer[1])
        cscf_model.load_weights(cscf_pretrain)        
        sscf_model,_ = SSCF().get_model(num_users, seq_len[2], num_items, num_factors, num_negatives, num_layer[2])
        sscf_model.load_weights(sscf_pretrain)
        
        model = load_pretrain_model(model, ascf_model, cscf_model, sscf_model,  num_layer, alpha)
        print("Load pretrained ASCF (%s) and CSCF (%s) and SSCF (%s) models done. " %(ascf_pretrain, cscf_pretrain, sscf_pretrain))
    
    # Init performance
    t1 = time()
    (hits, ndcgs) = evaluate_scf(model, u_model, train, testRatings, testNegatives, topK, max(seq_len), num_items)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f\t [%.1f s]' % (hr, ndcg, time()-t1))
    
    
    # Train model
    best_hr, best_ndcg, best_iter = hr, ndcg, 0
    if args.out > 0:
        model_out_file = 'Pretrain/%s_uscf_%d_%s_%s_%d_%.4f_%.4f.h5' %( args.dataset, 0, seq_len, num_layer, num_negatives, best_hr, best_ndcg)
        model.save_weights(model_out_file, overwrite=True)
        
    for epoch in range(1, 1+epochs):
        t0 = time()
        # Generate training instances
        user_input, x_input, session_input, item_input, labels = get_train_instances(train, num_negatives, max(seq_len), num_items)
        t1 = time()
        # Training
        hist = model.fit([np.array(user_input), np.array(x_input),np.array(session_input), np.array(item_input)], #input
                         np.array(labels), # labels 
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        t2 = time()
        
        # Evaluation
        if epoch %verbose == 0:
            (hits, ndcgs) = evaluate_scf(model, u_model, train, testRatings, testNegatives, topK, max(seq_len), num_items)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s, %.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                  % (epoch,  t2-t1, t1-t0, hr, ndcg, loss, time()-t2))
            if (hr+ndcg) > (best_hr+best_ndcg):
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    model_out_file = 'Pretrain/%s_uscf_%d_%s_%s_%d_%.4f_%.4f.h5' %( args.dataset, epoch, seq_len, num_layer, num_negatives, best_hr, best_ndcg)
                    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best model is saved to %s" %(model_out_file))