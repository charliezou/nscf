# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 12:01:49 2018

@author: charliezou
"""
from keras import backend as K
from keras.models import Model
from keras.layers import Lambda, Embedding, Input, Add

from commonlayer import create_n_attnlayer
from framework import SCF_Framework

class ASCF(SCF_Framework):
    def __init__(self, **kwargs):
        super().__init__('ASCF', **kwargs)          
    
    def get_model(self, num_users, seq_len, num_items, latent_dim, num_negatives, num_layer, gt=1):
        # Input variables
        u_input = Input(shape=(1,), dtype='int32', name = 'u_input')
        x_input = Input(shape=(seq_len,), dtype='int32', name = 'x_input')
        session_input = Input(shape=(seq_len,), dtype='int32', name = 'session_input')    
        item_input = Input(shape=(num_negatives+1,), dtype='int32', name = 'item_input')
        
        if gt > 0:        
            T_Embedding_GT = Embedding(input_dim = num_users, output_dim = latent_dim, name = 't_gt_embedding')        
        K_Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_dim, name = 'k_item_embedding')
        Q_Embedding_Item = Embedding(input_dim = 1, output_dim = latent_dim, name = 'q_item_embedding')
        K_Embedding_Session = Embedding(input_dim = seq_len, output_dim = latent_dim, name = 'session_embedding')         
        T_Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_dim, name = 'item_embedding')     
        
        q_input =Lambda(lambda x: K.zeros((K.shape(x)[0], 1), dtype='int32'))(x_input)
        q_latent = Q_Embedding_Item(q_input)
            
        k_latent = K_Embedding_Item(x_input)      
        k_latent = Add()([k_latent, K_Embedding_Session(session_input)])
        
        user_latent, _ = create_n_attnlayer(latent_dim, num_items, num_layer, x_input, q_latent, k_latent)  
        if gt > 0:      
            user_latent = Add()([user_latent, T_Embedding_GT(u_input)])
          
        item_latent = T_Embedding_Item(item_input)
            
        prediction = Lambda(lambda x: K.sigmoid(K.batch_dot(x[0][:,0,:], x[1],axes=(1,2))))([user_latent, item_latent])
        
        user_vec_model = Model(inputs=[u_input, x_input, session_input], outputs=user_latent)    
        model = Model(inputs=[u_input, x_input, session_input, item_input], outputs=prediction)
        
        model.summary()
                
        return model, user_vec_model    
    
if __name__ == '__main__':
    ascf = ASCF()
    ascf.execution()