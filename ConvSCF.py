# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 12:01:49 2018

@author: charliezou
"""
from keras import backend as K
from keras.models import Model
from keras.layers import Lambda, Embedding, Input, Add, Conv1D
from commonlayer import Position_Embedding

from framework import SCF_Framework

class CSCF(SCF_Framework):
    def __init__(self, **kwargs):
        super().__init__('CSCF', **kwargs)    
    
    def get_model(self, num_users, seq_len, num_items, latent_dim, num_negatives, num_layer, gt=1):
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
        k_latent = Position_Embedding()(k_latent) 
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
    
if __name__ == '__main__':
    cscf = CSCF()
    cscf.execution()