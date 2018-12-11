# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 10:40:22 2018

@author: charliezou

This code is referenced from the NeaFM model code in the github url (https://github.com/hexiangnan/neural_collaborative_filtering)
"""

class RatingItem(object):
    def __init__(self, itemList, scoreList, tList, negItems):
        self.items = itemList
        self.scores = scoreList
        self.ts = tList
        self.negItems = negItems
        
class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.trainRatings = self.load_rating_file_as_dict(path + ".train.rating")
        self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
        self.testNegatives = self.load_negative_file(path + ".test.negative")
        assert len(self.testRatings) == len(self.testNegatives)

    def load_rating_file_as_dict(self, filename):
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        self.num_users, self.num_items = num_users + 1,  num_items + 1
        
        allItems = [i for i in range(self.num_items)]
        ratingdict = {}
        baseuser = -1
        itemList = []
        scoreList = []
        timeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating, t  = int(arr[0]), int(arr[1]), float(arr[2]), int(arr[3])
                if (rating > 0):
                    if user != baseuser:
                        if baseuser != -1:                             
                            ratingdict[baseuser] = RatingItem(itemList, scoreList, timeList, list(set(allItems) - set(itemList)))
                            
                        baseuser = user
                        itemList, scoreList, timeList  = [item], [rating], [t]
                                                
                    else:
                        itemList.append(item)
                        scoreList.append(rating)
                        timeList.append(t)                        
                line = f.readline()                       
        ratingdict[baseuser] = RatingItem(itemList, scoreList, timeList, list(set(allItems) - set(itemList)))
        
        return ratingdict
    
    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList
    
    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList