import numpy as np
from numpy import matrix
from copy import copy


class HDM: #HDM memory model (see Kelly, Kwok, & West, 2015)

        #%actual values of these properties are set in the constructor
        percepts = 0
        concepts = 0
        labels = 0
        placeholder = 0
        left = 0
        n = 0 # n: the dimensionality of a percept or concept
        m = 0 # m: the number of percepts / concepts
        # activation threshold: if max similarity is below this, retrieval failure occurs
        activationThreshold = 0
        name = 'HDM';  # name of the model


        # construct HDM model
        def __init__(self, concepts, percepts, labels, name, placeholder, left):
            [self.m,self.n] = np.shape(concepts);

            # create the model
            self.concepts = copy(concepts)
            self.percepts = copy(percepts)
            self.labels = copy(labels)
            
            if not name is None:
                self.name = name            
            # the placeholder vector - acts as a question mark
            if not placeholder is None:
                self.placeholder = placeholder
            else:
                self.placeholder = normalVector(self.n)
            # left - a permutation indicating order
            if not left is None:
                self.left = left
            else:
                self.left = getShuffle(self.n);
        
        """ given probe, find memory vector most similar to probe
         retrieve:
           maxSim: similarity of that memory vector
           index: index of that memory vector
           percept: environment vector corresponding to that memory vector"""

        def Retrieve(self,probe):
            
            # find concept most similar to the probe
            similarities = self.Resonance(probe)
            [maxSim,index] = max(similarities)
            
            # here's the label and percept if you want it
            if maxSim > self.activationThreshold:
                label = self.labels(index)
                percept = self.percepts(index)
            else:
                # if nothing in memory has greater similarity than zero
                # the following default values are returned
                index = 0
                maxSim = self.activationThreshold
                label = 'retrieval failure'
                percept = np.array([0]*self.n)
            return maxSim,index,label,percept
                
        # calculate all similarities between probe and memory vectors
        def resonance(self,probe):
            similarities = np.array([0]*self.m)
            for i in range (0,self.m):
                similarities[i] = self.getSimilarity(i,probe)
            return similarities

        # get the similarity between the probe and indexed memory vector
        def getSimilarity(self,index,probe):
            item = self.concepts[index];
            if item is None:
                similarity = 0
            else:
                similarity = vectorCosine(item,probe)
            return similarity
        
        # add an experience (represented by a vector) of indexed concept
        def HDMadd(self,index,experience):
            #print experience, self.concepts[index]
            self.concepts[index] = np.add(self.concepts[index], experience)
            
        
        """ read in a set of sentences commit those associations to memory
         indices is a matrix of percept/concept indices
           each row of indices is a 'sentence' of co-occurring terms
           each column of indices is a sentence position """
        def read(self,indices):              
            [num_sentences, num_words] = np.shape(indices)
            for i in range(0,num_sentences):
                # check to see if there's a invalid index:
                #   disallow Inf, NaN, and Zero values
                foundMin=min(numpy.where(x == 0)[0],numpy.where(x == np.nan)[0],numpy.where(x == np.inf)[0],numpy.where(x == -np.inf)[0])
                # if there is an invalid index, end the sentence before it
                if foundMin[0]:
                    sent_len = foundMin[0] - 1
                else: # otherwise, the sentence is the maximum length
                    sent_len = num_words
                # construct the sentence
                sentence  = np.zeros(sent_len,self.n)
                for w in range (0,sent_len):
                    sentence[w] = self.percepts[indices[i,w]]
                # for each word in the sentence, construct a query
                # and convert the query into a vector using hdmUOG
                # then add that vector to memory
                for w in range (0,sent_len):
                    query = sentence
                    query[w] = self.placeholder  
                    self.add(indices[i,w],hdmUOG(query,w,self.left)) 
                
        
        # retrieve from HDM the memory vector identified by label
        def getConcept(self,label):
            index = lookup(label,self.labels)
            concept = self.concepts[index]
            return concept
        
        # retrieve from HDM the environment vector identified by label
        def getPercept(self,label):
            index = lookup(label,obj.labels)
            percept = self.percepts[index]
            return percept
    

