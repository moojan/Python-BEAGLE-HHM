
""" LOOKUP finds an exact match to string WORD in cell array DICTIONARY """

def lookup ( word, dictionary):
    if (not word in dictionary):
        return -1
    return dictionary.index(word)

