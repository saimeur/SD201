from typing import List

from PointSet import PointSet, FeaturesTypes

import numpy as np

class Tree:
    """A decision Tree

    Attributes
    ----------
        points : PointSet
            The training points of the tree
    """
    def __init__(self,
                  features: List[List[float]],
                  labels: List[bool],
                  types: List[FeaturesTypes],
                  h: int = 1,
                  min_split_points : int = 1,
                  beta : float = 0.0
                  ):
        """
        Parameters
        ----------
            labels : List[bool]
                The labels of the training points.
            features : List[List[float]]
                The features of the training points. Each sublist
                represents a single point. The sublists should have
                the same length as the `types` parameter, while the
                list itself should have the same length as the
                `labels` parameter.
            types : List[FeaturesTypes]
                The types of the features.
            h : height of the three (at 1 by default).
            min_split_points (cf question 9)
        """
        self.beta = beta
        self.points = PointSet(features, labels, types)
        self.min_split_points = min_split_points
        self.points.set_min_split_points(min_split_points) 
        self.h =h
        (ind, g) = self.points.get_best_gain()
        k= self.points.k
        threshlod = self.points.threshold
        self.compt = 0
        self.root = (ind, k, threshlod) #The root contain feature's id and the categorical's id of the split and the threshold
        self.build_tree()
        #We check if it's a leaf or if is not relevent to split the PointSet anymore
        '''if h==0 or ind == None :
            self.left_child = None
            self.right_child = None
            self.h = 0
        
        else :
            #Same as get_categorical_best_gini, the left child VALIDATES THE CONDITION OF THE SPLIT and the right child invalidates it
            if self.points.types[ind] == FeaturesTypes.BOOLEAN:
                l = PointSet(self.points.features[self.points.features[:,ind]==1], self.points.labels[self.points.features.T[ind]==1], self.points.types)
                r = PointSet(self.points.features[self.points.features.T[ind]==0], self.points.labels[self.points.features.T[ind]==0], self.points.types)
                self.left_child = Tree(l.features,l.labels,types,self.h-1,min_split_points)
                self.right_child = Tree(r.features,r.labels,types,self.h-1,min_split_points)
            elif self.points.types[ind] == FeaturesTypes.CLASSES :
                l = PointSet(self.points.features[self.points.features[:,ind]==k], self.points.labels[self.points.features.T[ind]==k], self.points.types)
                r = PointSet(self.points.features[self.points.features.T[ind]!=k], self.points.labels[self.points.features.T[ind]!=k], self.points.types)
                self.left_child = Tree(l.features,l.labels,types,self.h-1,min_split_points)
                self.right_child = Tree(r.features,r.labels,types,self.h-1,min_split_points)
            else:
                l = PointSet(self.points.features[self.points.features[:,ind]<=threshlod], self.points.labels[self.points.features.T[ind]<=threshlod], self.points.types)
                r = PointSet(self.points.features[self.points.features.T[ind]>threshlod], self.points.labels[self.points.features.T[ind]>threshlod], self.points.types)
                self.left_child = Tree(l.features,l.labels,types,self.h-1,min_split_points)
                self.right_child = Tree(r.features,r.labels,types,self.h-1,min_split_points)'''
    
    def reset_compteur(self):
        self.compt = 0
        if self.h==0:
            return None
        else:
            self.right_child.reset_compteur()
            self.left_child.reset_compteur()
        

    def decide(self, features: List[float]) -> bool:
        """Give the guessed label of the tree to an unlabeled point

        Parameters
        ----------
            features : List[float]
                The features of the unlabeled point.

        Returns
        -------
            bool
                The label of the unlabeled point,
                guessed by the Tree
        """
        #We define decide in a recursive way

        (ind, k, threshlod)= self.root
        #We test the stoping condition
        if self.h ==0 :
            return self.points.labels.mean()>0.5
        else:
            if k != None:
                #We have categorical type
                if features[ind]==k:
                    return self.left_child.decide(features)
                else:
                    return self.right_child.decide(features)
            elif threshlod != None:
                if features[ind]<=threshlod:
                    return self.left_child.decide(features)
                else:
                    return self.right_child.decide(features)
            else :
                if features[ind]==1:
                    return self.left_child.decide(features)
                else:
                    return self.right_child.decide(features)
                
        raise NotImplementedError('Implement this method for Question 4')

    def add_training_point(self, features: List[float], label: bool) -> None :
        (ind, k, threshlod)= self.root
        self.compt +=1
        self.points.add_sample(features, label)
        if self.compt >= self.beta* len(self.points.features):
            print('yes')
            self.reset_compteur()
            lo = self.points.features
            self.build_tree()
            print(np.array_equal(lo,self.points.features ))
        else:
            if self.h ==0 :
                return None
            else:
                if k != None:
                    #We have categorical type
                    if features[ind]==k:
                        return self.left_child.decide(features)
                    else:
                        return self.right_child.decide(features)
                elif threshlod != None:
                    if features[ind]<=threshlod:
                        return self.left_child.decide(features)
                    else:   
                        return self.right_child.decide(features)
                else :
                    if features[ind]==1:
                        return self.left_child.decide(features)
                    else:
                        return self.right_child.decide(features)
                    
    def del_training_point(self, features: List[float], label: bool) -> None :
        (ind, k, threshlod)= self.root
        self.compt +=1
        self.points.del_sample(features, label)
        if self.compt >= self.beta* len(self.points.features):
            self.reset_compteur()
            arbre = Tree(self.points.features, self.points.labels, self.points.types, self.h, self.min_split_points)
            lo =self.points.features
            self.build_tree()
            print(np.array_equal(lo,self.points.features ))
        else:
            if self.h ==0 :
                return None
            else:
                if k != None:
                    #We have categorical type
                    if features[ind]==k:
                        return self.left_child.decide(features)
                    else:
                        return self.right_child.decide(features)
                elif threshlod != None:
                    if features[ind]<=threshlod:
                        return self.left_child.decide(features)
                    else:   
                        return self.right_child.decide(features)
                else :
                    if features[ind]==1:
                        return self.left_child.decide(features)
                    else:
                        return self.right_child.decide(features)

    def build_tree(self):
        (ind, k, threshlod)=self.root
        #We check if it's a leaf or if is not relevent to split the PointSet anymore
        if self.h==0 or ind == None :
            self.left_child = None
            self.right_child = None
            self.h = 0
        
        else :
            #Same as get_categorical_best_gini, the left child VALIDATES THE CONDITION OF THE SPLIT and the right child invalidates it
            if self.points.types[ind] == FeaturesTypes.BOOLEAN:
                l = PointSet(self.points.features[self.points.features[:,ind]==1], self.points.labels[self.points.features.T[ind]==1], self.points.types)
                r = PointSet(self.points.features[self.points.features.T[ind]==0], self.points.labels[self.points.features.T[ind]==0], self.points.types)
                self.left_child = Tree(l.features,l.labels,self.points.types,self.h-1,self.min_split_points)
                self.right_child = Tree(r.features,r.labels,self.points.types,self.h-1,self.min_split_points)
            elif self.points.types[ind] == FeaturesTypes.CLASSES :
                l = PointSet(self.points.features[self.points.features[:,ind]==k], self.points.labels[self.points.features.T[ind]==k], self.points.types)
                r = PointSet(self.points.features[self.points.features.T[ind]!=k], self.points.labels[self.points.features.T[ind]!=k], self.points.types)
                self.left_child = Tree(l.features,l.labels,self.points.types,self.h-1,self.min_split_points)
                self.right_child = Tree(r.features,r.labels,self.points.types,self.h-1,self.min_split_points)
            else:
                l = PointSet(self.points.features[self.points.features[:,ind]<=threshlod], self.points.labels[self.points.features.T[ind]<=threshlod], self.points.types)
                r = PointSet(self.points.features[self.points.features.T[ind]>threshlod], self.points.labels[self.points.features.T[ind]>threshlod], self.points.types)
                self.left_child = Tree(l.features,l.labels,self.points.types,self.h-1,self.min_split_points)
                self.right_child = Tree(r.features,r.labels,self.points.types,self.h-1,self.min_split_points)
        
