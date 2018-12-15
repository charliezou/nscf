# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 10:40:22 2018

@author: charliezou

Some of the code is referenced from https://github.com/hexiangnan/neural_collaborative_filtering 

"""

import math
import heapq # for retrieval topK
import numpy as np
from commonlayer import get_session_list

def evaluate_scf(model, u_model, trainRatings, testRatings, testNegatives, K, seq_len, num_items):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """    
    item_embedding = model.get_layer('item_embedding').get_weights()[0]    
        
    hits, ndcgs = [],[]  
    
    # Single thread
    for idx in range(len(testRatings)):
        rating = testRatings[idx]
        items = testNegatives[idx]
        u_input = rating[0]            
        x_input = [trainRatings[u_input].items[:seq_len]]        
        session_input =[get_session_list(trainRatings[u_input].ts[:seq_len])]
            
        gtItem = rating[1]
        items.append(gtItem)
        # Get prediction scores
            
        user_vec = u_model.predict([np.asarray([[u_input]]), np.asarray(x_input) ,np.asarray(session_input)])[0, 0,:]    
        item_vec = item_embedding[items]    
        predictions = np.dot(item_vec, user_vec)
        map_item_score = {}
        for i in range(len(items)):
            item = items[i]
            map_item_score[item] = predictions[i]    
        
        # Evaluate top rank list
        ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)
        hr = getHitRatio(ranklist, gtItem)
        ndcg = getNDCG(ranklist, gtItem)
        
        hits.append(hr)
        ndcgs.append(ndcg)      
    return (hits, ndcgs)


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0
