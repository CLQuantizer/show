import numpy as np
# sentence manipulating tools
def find_sentence_that_have(key, sentences):
    sentence_list = []
    for sentence in sentences: 
        if key in sentence and sentence not in sentence_list:
            sentence_list.append(sentence)
    return sentence_list

def comparative_sent_printing(senlist1,senlist2):
    assert len(senlist1)==len(senlist2)
    for i in range(len(senlist1)):
        print(f'{i}th pair:')
        print(senlist1[i])
        print(senlist2[i])

def x2y(ss,x,y):
    modified=[]
    for s in ss:
        word_list =[]
        for i,word in enumerate(s):
            if word == x:
                word_list.append(y)
            else:
                word_list.append(word)
        modified.append(word_list)
    return modified