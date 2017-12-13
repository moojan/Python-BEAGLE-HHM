
"""Reads labels and sentences from the file. Each row represents a sentence. 
   The first index in each row is the length of the sentence, 
   and the rest are indexes of the words. Lables has the information about the words.
   A cheating word is added in the beginning, so the indexes actually represent the word number in the sentence.
   This function returns the matrix of sentences and the lables vector.
   Because sentence length is different, a maximum sentence length is set in the beginning
   and then the remaining indexes of the row are filled with -1s.
"""

import csv
import numpy as np

def readCorpus():
    corpus=[]
    MAX_SENTENCE_LENGTH=80 ## CHANGE THIS BASED ON THE CORPUS
    corpus = [line.rstrip('\n') for line in open("./data/swbd.txt_INDEX.txt")]
    index=0
    matrix=np.zeros([len(corpus), MAX_SENTENCE_LENGTH])
    for line in corpus:
        sentence=line.lstrip()
        parsed = np.array([int(n) for n in sentence.split()])
        matrix[index]=np.hstack((parsed,np.array((MAX_SENTENCE_LENGTH-len(parsed))*[-1])))
        index=index+1   
    label= [line.rstrip('\n') for line in open("./data/swbd.txt_WORDS.txt")]
    labels=[]
    labels.append("theCheatingWord!") ## To take care of 0 indexing, so the indexes actually represent the word number in the sentence.
    for line in label:
        labels.append(line.split(" ")[0])
    print matrix[0], labels[0], labels[1], len(labels), np.shape(matrix)
    print "Read data from the corpus"
    return labels, matrix


# For test purpuses:
#labels, matrix=readCorpus()
#print len(labels), np.shape(matrix)




