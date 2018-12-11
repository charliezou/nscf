# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 11:19:46 2018

@author: charliezou
"""

from keras.layers import Layer, Embedding, Add
import keras.backend as K
from keras.layers import Dropout


class Position_Embedding(Layer):    
    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size #须为偶数
        self.mode = mode
        super().__init__(**kwargs)
        
    def call(self, x):
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        position_j = 1. / K.pow(10000., \
                                 2 * K.arange(self.size / 2, dtype='float32' \
                               ) / self.size)
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(x[:,:,0]), 1)-1 #K.arange不支持变长，只好用这种方法生成
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)
        
    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2]+self.size)

#Single head attention         
class SingleAttention(Layer):
    def __init__(self, attn_dropout=None, **kwargs):
        super().__init__(**kwargs)
        self.attn_dropout = attn_dropout

    def build(self, input_shape):
        self.size_head = input_shape[0][-1]
        super().build(input_shape)
                     
    def call(self, x):
        Q_seq,K_seq,V_seq = x        
        A = K.batch_dot(Q_seq, K_seq, axes=[2,2]) / self.size_head**0.5
        A = K.softmax(A)
        if self.attn_dropout is not None:
            A = Dropout(self.attn_dropout)(A)
        O_seq = K.batch_dot(A, V_seq, axes=[2,1])
        return O_seq
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[2][2]) 
     

class EncoderLayer:
    def __init__(self, embedding_dim: int, n_size: int , layer_id: int, attn_dropout = None):
        self.embedding_dim = embedding_dim
        self.n_size = n_size
        self.layer_id = layer_id
        self.embdedding = Embedding(input_dim = n_size, output_dim = embedding_dim, name ='v_embedding_layer_{}'.format(layer_id))  
        self.attention = SingleAttention(attn_dropout, name = 'attn_layer_{}'.format(layer_id)) 

    def __call__(self, x, q_seq, k_seq):
        v_seq = self.embdedding(x)        
        o_seq = self.attention([q_seq, k_seq, v_seq])
        return o_seq, v_seq 
  

def create_n_attnlayer(embedding_dim: int, n_size: int, num_layers: int, x, q_seq, k_seq, attn_dropout=None):
    k_seq = Position_Embedding()(k_seq)
    
    for i in range(num_layers):
        q_seq_t, k_seq_t = EncoderLayer(embedding_dim, n_size, i, attn_dropout)(x, q_seq, k_seq)
        q_seq = Add()([q_seq, q_seq_t]) if i>0 else  q_seq_t      
        k_seq = Add()([k_seq, k_seq_t]) if i>0 else  k_seq_t 

    return q_seq, k_seq

#to generate session id     
def get_session_list(ts):
    s=0
    t=-1
    sessionList = []
    for i in range(len(ts)):
        if i==0:
            t = ts[i]
        elif t != ts[i]:
            s = s+1
            t = ts[i]
        sessionList.append(s)
    return sessionList   