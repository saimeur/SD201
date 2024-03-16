from typing import List, Tuple

from enum import Enum
import numpy as np

class FeaturesTypes(Enum):
    """Enumerate possible features types"""
    BOOLEAN=0
    CLASSES=1
    REAL=2

class PointSet:
    """A class representing set of training points.

    Attributes
    ----------
        types : List[FeaturesTypes]
            Each element of this list is the type of one of the
            features of each point
        features : np.array[float]
            2D array containing the features of the points. Each line
            corresponds to a point, each column to a feature.
        labels : np.array[bool]
            1D array containing the labels of the points.
    """
    def __init__(self, features: List[List[float]], labels: List[bool], types: List[FeaturesTypes]):
        """
        Parameters
        ----------
        features : List[List[float]]
            The features of the points. Each sublist contained in the
            list represents a point, each of its elements is a feature
            of the point. All the sublists should have the same size as
            the `types` parameter, and the list itself should have the
            same size as the `labels` parameter.
        labels : List[bool]
            The labels of the points.
        types : List[FeaturesTypes]
            The types of the features of the points.
        """
        self.types = types
        self.features = np.array(features)
        self.labels = np.array(labels)
        self.k = None #Reprensent the category of the split if it split on a class
        self.threshold = None #Reprensent the threshold of the split if it split on a real
        self.min_split_points = 1
    
    def set_min_split_points(self, x :int):
        self.min_split_points = x
    
    def add_sample(self,feat, label):
        self.features = np.vstack([self.features, feat])
        self.labels = np.append(self.labels, label)

    def del_sample(self,feat,label):
        index_list = []
        for i in range(len(self.features)):
            if np.array_equal(self.features[i],feat):
                index_list.append(i)
        for j in index_list:
            np.delete(self.features, j, axis=0)
            np.delete(self.labels, j, axis=0)
    
    def get_gini(self) -> float:
        """Computes the Gini score of the set of points

        Returns
        -------
        float
            The Gini score of the set of points
           """
        #We need the gini gain to be 0 for an empty set for other questions
        if len(self.features)== 0:
            return 0
        
        compt_class1 = 0 ; compt_class2 = 0
        for x in self.labels:
            if x : compt_class1 += 1 
            else : compt_class2+=1
        
        total = compt_class2 + compt_class1
        
        return 1- (compt_class1/total)**2 - (compt_class2/total)**2



        raise NotImplementedError('Please implement this function for Question 1')
        
    #This function is an evolution of the function above that can compute for features with Boolean type but also categorical type
    def get_best_gain(self) -> Tuple[int, float]:
        """Compute the feature along which splitting provides the best gain

        Returns
        -------
        int
            The ID of the feature along which splitting the set provides the
            best Gini gain.
        int 
            The ID of the category of the features along which splitting the set provides the
            best Gini gain (None if we split on a boolean).
        float
            The best Gini gain achievable by splitting this set along one of
            its features.
        """
        giniParents = self.get_gini()
        n = len(self.features)
        max=0
        res = (None,None) #We initialise the result
        #For each features
        for i in range(self.features.shape[1]): 
            #We test if it's a boolean and if it is we do compute the gini gain by spliting on index i
            if self.types[i] == FeaturesTypes.BOOLEAN:
                a= PointSet(self.features[self.features.T[i]==1], self.labels[self.features.T[i]==1], self.types)
                b = PointSet(self.features[self.features.T[i]==0], self.labels[self.features.T[i]==0], self.types)
                gini_gain = giniParents - len(a.features)*a.get_gini()/n - len(b.features)*b.get_gini()/n
                if gini_gain > max and len(a.features)>= self.min_split_points and len(b.features)>= self.min_split_points:
                    max = gini_gain
                    res = (i, gini_gain)
            #We do the same for the type CLASSES
            elif self.types[i] == FeaturesTypes.CLASSES:
                #We have a categorical type
                values = np.unique(self.features[:,i]) #All disctinct possible categories
                #For each values possible we split on : features == values and features != values
                for k in values:
                    a= PointSet(self.features[self.features.T[i]==k], self.labels[self.features.T[i]==k], self.types)
                    b = PointSet(self.features[self.features.T[i]!=k], self.labels[self.features.T[i]!=k], self.types)
                    gini_gain = giniParents - len(a.features)*a.get_gini()/n - len(b.features)*b.get_gini()/n
                    if gini_gain > max and len(a.features)>= self.min_split_points and len(b.features)>= self.min_split_points:
                        max = gini_gain
                        res = (i, gini_gain)
                        self.k = k
            #We do the same for the type REAL
            else:
                #We have a continuous type
                values = np.unique(self.features[:,i]) #All disctinct possible categories
                #For each values possible we split on : features == values and features != values
                for k in values:
                    a= PointSet(self.features[self.features.T[i]>=k], self.labels[self.features.T[i]>=k], self.types)
                    b = PointSet(self.features[self.features.T[i]<k], self.labels[self.features.T[i]<k], self.types)
                    gini_gain = giniParents - len(a.features)*a.get_gini()/n - len(b.features)*b.get_gini()/n
                    if gini_gain > max and len(a.features)>= self.min_split_points and len(b.features)>= self.min_split_points:
                        max = gini_gain
                        res = (i, gini_gain)
                        self.threshold = (min(a.features[:,i])+ np.max(b.features[:,i]) )/2
        #k != None iff we split on a classes
        if res[0] == None or self.types[res[0]] != FeaturesTypes.CLASSES:
            self.k = None
        #threshold != None iff we split on a real
        if res[0] == None or self.types[res[0]] != FeaturesTypes.REAL:
            self.threshold = None
        return res

        raise NotImplementedError('Please implement this methode for Question 6')
    
    def get_best_threshold(self) -> float:
        if self.k != None:
            return self.k
        else:
            return self.threshold


