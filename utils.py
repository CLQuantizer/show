import numpy as np
from collections import defaultdict
import random
import time as T
import matplotlib.pyplot as plt
count_sign = '__COUNT__'

# data split function
def train_validation_test_split(data, train_percent, validation_percent):
    """
    Splits the input data to  train/validation/test according to the percentage provided
    
    Args:
        data: Pre-processed and tokenized corpus, i.e. list of sentences.
        train_percent: integer 0-100, defines the portion of input corpus allocated for training
        validation_percent: integer 0-100, defines the portion of input corpus allocated for validation
        
        Note: train_percent + validation_percent need to be <=100
              the reminder to 100 is allocated for the test set
    
    Returns:
        train_data: list of sentences, the training part of the corpus
        validation_data: list of sentences, the validation part of the corpus
        test_data: list of sentences, the test part of the corpus
    """
    # fixed seed here for reproducibility
    random.seed(2020)
    
    # reshuffle all input sentences
    random.shuffle(data)

    train_size = int(len(data) * train_percent / 100)
    train_data = data[0:train_size]
    
    validation_size = int(len(data) * validation_percent / 100)
    validation_data = data[train_size:train_size + validation_size]
    
    test_data = data[train_size + validation_size:]
    
    return train_data, validation_data, test_data

def single_pass_trigram_count(sentence_list, count_sign = '__COUNT__'):
    """
    Creates the Ngram count matrix from the input corpus in a single pass through the corpus.
    Args:
        corpus: Pre-processed and tokenized corpus. 
        
    Returns:
       trie
    """
    vocabulary = []
    trie = {}

    for sentid, each in enumerate(sentence_list):
        if len(each)<2:
            continue
            
        assert type(each)==list 
    # Pad each sentence with N-1 starting token <s> at the beginning and one end token </s> at last. 
        padded_sentence = ['<s>' for _ in range(3-1)] + each + ['</s>']

    # go through the sentence once with a sliding window of size N
        for i in range(len(padded_sentence) - 3 + 1):
        # the sliding window starts at position i and contains N words
            trigram = padded_sentence[i : i + 3]
#             print(trigram)
            Node0 = trigram[0]
            Node1 = trigram[1]
            Node2 = trigram[2]
            
            if Node0 not in trie.keys():
                trie[Node0] = {count_sign:1}
            else:
                trie[Node0][count_sign]+=1
                
            if Node1 not in trie[Node0].keys():
                trie[Node0][Node1] = {count_sign:1}
            else:
                trie[Node0][Node1][count_sign]+=1
                
            if Node2 not in trie[Node0][Node1].keys():
                trie[Node0][Node1][Node2] = {count_sign:1}
            else:
                trie[Node0][Node1][Node2][count_sign]+=1

        #  Checkpoint
        if sentid>0 and sentid%5000==0:
            print(f'{sentid} sentences have been processed. Now we\'re at {padded_sentence}.')
            print(f'There are now {len(trie)} vocabs in the dictionary.')
        if sentid==(len(sentence_list)-1):
            print(f'-------------------------Single pass for trigram finished!--------------------------------')
        
    rootcount=0
    for key in trie.keys():
        rootcount+=trie[key][count_sign]
    trie[count_sign] = rootcount
             
    return trie

def show_unigram_prob(model, unigram):
    count = 0
    for key in model.keys():
        if key != count_sign:
            count += model[key][count_sign]
    
    prob =  round( np.log(model[unigram][count_sign]/count) , 4)
#     rooprint(f'logprob of ---->{unigram}<----- is {prob}')
    return prob

def AssignProbSent(sentence, trie, count_sign = '__COUNT__', alpha = 0.4): 
    
    
    '''
    Arguments: sentence = list of str, trie model= trie of ngrams, count_sign as a default parameter
    Returns: probability of any sentence based on the model 

    '''
    
    def Assign_prob_trigram(trigram, trie):

        '''
        Arguments: sentence = list of str with length 3, trie model= trie of ngrams, count_sign
        Returns: probability of any sentence based on the model 
        '''
        def in_trie(sequence, trie, path):
            '''
            decide if a word is in a dictionary trie
            '''
            if type(sequence) == list:
                new_path = path
                word = sequence[0]
            elif type(sequence) == str:
                new_path = path
                word = sequence
            else:
                return 'Wrong input type'
#             print('sequence is',sequence)
            if word in trie.keys():
#                 print(f'now searching {len(trie[word])}')
                new_path.append(word)
#                 print('path is:', new_path)
        #         print('length is: ', len(sequence))
                if len(sequence)==1 or type(sequence)==str:
#                     print('length is now 1, ready to exit')
#                     print(new_path)
                    return new_path
                else:
                    return in_trie(sequence[1:], trie[word], new_path)
            else:
#                 print(f'{word} not in trie')
                return False

        assert len(trigram)==3, 'No, this ain\'t a trigram that we\'re looking at'
        
        estimation_space =  0.05 * len(trie)
        
        x = in_trie(trigram, trie,[])
        if x:
            # if trigram exists
            self_count = trie[x[0]][x[1]][x[2]][count_sign]
            parent_count =  trie[x[0]][x[1]][count_sign]
            return np.log(self_count)-np.log(parent_count)
        else:  
            # if trigram does not exist but BIGRAM-prefix exist
#             print(f'trigram {trigram} does not exist but its bigram-prefix {trigram[:2]} does: ')
            y = in_trie(trigram[:2],trie,[])
            if y:
                C2=1
                for key in trie[y[0]][y[1]].keys():
                    if key!= count_sign:
                        C2 += (trie[y[0]][y[1]][key][count_sign]==1)
               
    #             print('parent count:  ',trie[y[0]][y[1]][count_sign])
                parent_count = trie[y[0]][y[1]][count_sign]
            
#                 print('parentcount is:', parent_count)
#                 print('effective count is: ', C2/parent_count)
                return np.log(C2)-np.log(parent_count)-np.log(estimation_space)
            else:
    #             if tri does not but LATTER bigram exist:
                z = in_trie(trigram[1:],trie,[])
                if z:
#                     print('bigram bacukoff:', trigram[1]+' '+trigram[2])
                    self_count = trie[z[0]][z[1]][count_sign]
                    parent_count =  trie[z[0]][count_sign]
                    return np.log(self_count)-np.log(parent_count)+np.log(alpha)
                else:
    #                if bigram does not exist 
                    zz = in_trie(trigram[1],trie,[])
                    if zz:
    #                  but  if the prefix of bigram exists
#                         print('no, trigram, relying bigram + unk: ', trigram[1],trigram[2])
                        C1 = 1
                        for key in trie[zz[0]].keys():
                            if key!= count_sign:
                                C1 += (trie[zz[0]][key][count_sign]==1)
                        parent_count = trie[zz[0]][count_sign] 
                            
                        return np.log(C1)-np.log(parent_count)-np.log(estimation_space*10)+np.log(alpha)
                    else:
    #                   if the prefix of bigram is an unknown word
#                         print('last word is ', trigram[2])
                        zzz = in_trie(trigram[2],trie,[])
                        if zzz:
#                             print('no, bigram, relying on unigram : ',trigram[2])
#                             print('last word is ',zzz)
            #                 if last word is in the vocab
#                             print('unigram only')
                            self_count = trie[zzz[0]][count_sign]
                            parent_count = trie[count_sign]
                            return np.log(self_count)-np.log(parent_count)+2*np.log(alpha)
                        else:
                            print(f'{trigram[1]} and {trigram[2]} are 2 consecutive unk word in {trigram}')
    #                     if this word is unknown
                          #   Good turing
                            C0 = 1
                            for key in trie.keys():
                                if key!=count_sign:
                                    C0 += (trie[key][count_sign]==1)
                            parent_count = trie[count_sign]
#                             print(trigram,' sharpe down', np.log(unk0*alpha*alpha))
                            return np.log(C0)-np.log(parent_count) - np.log(estimation_space*20) + np.log(alpha)*2
                        
#    begin sentence level processing

    padded_sentence = ['<s>']*2 + sentence + ['</s>']
    #sliding through each word
    log_prob_array = []
    for i in range(len(padded_sentence) - 3):
     
        trigram = padded_sentence[i:i+3]

        if i == 0:
            log_prob = Assign_prob_trigram(trigram,trie)
            log_prob_array.append(log_prob)
#             print(f'length is {len(log_prob_array)}')
#             print(log_prob_array[-1])
        else:
#             print(trigram)
            log_prob = Assign_prob_trigram(trigram, trie)
            prev_log_prob = log_prob_array[-1]
            log_prob_array.append(log_prob+prev_log_prob)
    return log_prob_array


def calculating_perplexity(eval_set, trie, AssignProbSent=AssignProbSent):
    perplexity_list=[]
    for i,sentence in enumerate(eval_set):
        M = len(sentence)
        p = AssignProbSent(sentence,trie)[-1]
# print(p, 'and')
#         print(f'probability is: {p}')
        perp =np.exp(p * (-1/M))
        perplexity_list.append(perp)
#         print(f'perplexity of sentence {i} is: {perp}')
        
#         if perp > 300:
#             print(f'Perplexity too high ({perp}): ',sentence)
        
#         if i == 0:
#             T_PREV = T.time()
        
#         if i % 1000==0:
#         if i % int(0.1*len(eval_set))==0 and i>0:
#             print(f'we\'re looking at number {i} in eval_set.\nIt is {sentence}')
#             T_END = T.time()
#             print(f'{0.1*len(eval_set)} sentences evaluated in {round(T_END - T_PREV,1)} seconds.')
#             T_PREV = T_END
                   
    mean = round(np.array(perplexity_list).mean(),1)
    print(f'The mean perplexity of these {len(perplexity_list)} sentences are {mean}')
    return mean

# visually verify the implementation of probability assignment algorithm on minimal pairs
def plot_minimal_pair(goodsent, badsent, model, assign_sentence_prob, __plt__=plt):

    f = assign_sentence_prob
    fig = plt.figure()
    fig.set_size_inches(15, 10)
    ax = fig.add_subplot(211)
    ay = fig.add_subplot(212)

    ax.set_xlabel('words')
    ax.set_ylabel('probability')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    ay.set_xlabel('words')
    ay.set_ylabel('possibility')
    ay.spines['bottom'].set_color('white')
    ay.spines['top'].set_color('white')
    ay.xaxis.label.set_color('white')
    ay.yaxis.label.set_color('white')
    ay.tick_params(axis='x', colors='white')
    ay.tick_params(axis='y', colors='white')

    x1_sen = []
    for i,word in enumerate(goodsent):
        x1_sen.append(str(i)+' '+word)
    x2_sen =[]
    for i,word in enumerate(badsent):
        x2_sen.append(str(i)+' '+word)

    prob1 = f(goodsent, model)
    ax.plot(x1_sen, prob1)
    for i,j in zip(x1_sen, prob1):
        ax.annotate(str(round(j,2)),xy=(i,j))

    prob2 = f(badsent, model)
    ay.plot(x2_sen, prob2)
    for i,j in zip(x2_sen, prob2):
        ay.annotate(str(round(j,2)),xy=(i,j))
    
    plt.show()
                                                            
    print('================\t===============\t===============\t===============\t===============\t')
    print()
    print(f'This is a plot illustrating how the probability of the test sentence: \n\n {goodsent}\n')
    print(f'and its miminal pair: \n\n{badsent} \n\ndeclines as the the algorithm process each word')
    return


def compare_sets(sentences,bad_sentences, model):
    
    def CompareProb(sentence, bad_sentence, model, index):
        Prob1 = AssignProbSent(sentence,model)[-1]
        Prob2 = AssignProbSent(bad_sentence,model)[-1]
#         print(f'Original sentence prob is {Prob1}')
#         print(f'Bad sentence praob is {Prob2}')
        if Prob1 >= Prob2:
            diff = Prob1- Prob2
            if diff< 4.7:
#                 print('Difference small')
#                 print(sentence)
                pass
            if diff> 18:
#                 print('Difference bibibibibibibibibibibibibibibig')
#                 print(sentence)
                pass
#             print(f'No.{index+1} bad sentence has lower prob, (by {Prob1-Prob2})')
        else:
            pass
#             print(f'bad sentence has higher logprob (by {Prob2-Prob1})')
#             print(sentence)
#             print(bad_sentence)
#             print()
        length = len(sentence)
        return Prob1, Prob2, length
    
    good_log_probs = np.zeros(len(sentences))
    bad_log_probs = np.zeros(len(sentences))
    good_perps = np.zeros(len(sentences))
    bad_perps = np.zeros(len(sentences))
    
    num = 0
    index = []
    
    for i,(a, b) in enumerate(zip(sentences, bad_sentences)):
#         print(f'\n{i}th pair')
        good_log_probs[i], bad_log_probs[i], M = CompareProb(a,b, model, i)
        good_perps[i] = np.exp(good_log_probs[i]*(-1/M))
        bad_perps[i] = np.exp(bad_log_probs[i]*(-1/M))
        
        if good_log_probs[i] >= bad_log_probs[i]:
            num+=1
        else:
            index.append(i)
    
    c = round(np.mean(good_log_probs),2)
    print(f'Good avg log_prob is : {round(np.mean(good_log_probs),2)}')
    print(f'Bad avg log_prob is : {round(np.mean(bad_log_probs),2)}')
    print('\n-----------------PERPLEXITY-------------------\n')
    print(f'good Perplexity is : {round(np.mean(good_perps),2)}')
    print(f'bad Perplexity is : {round(np.mean(bad_perps),2)}')
    
    print(f'{num} out of {len(sentences)} are correct predictions')
    return index

def log_uni_unk_prob(model):
    C0 = 0
    for key in model.keys():
        if key!=count_sign:
            C0 += (model[key][count_sign]==1)
    return np.log(C0)-np.log(model[count_sign])*2
            
def sentences_unigram_probability(sentences, model,log_uni_unk):
    def sentence_unigram_probability(sentence, model, log_uni_unk):
    #       
        log_prob_list = []
        for word in sentence:
            if word in model.keys():
                prob = np.log(model[word][count_sign]/model[count_sign])
                log_prob_list.append(prob)
            else:
                log_prob_list.append(log_uni_unk)
        mean = np.mean(log_prob_list)
        return mean 
    
    sentences_mean = []
    for s in sentences:
        sentences_mean.append(sentence_unigram_probability(s, model, log_uni_unk))
    return np.mean(sentences_mean)


    # **Calculating the perplexity using the evaluation dataset**
# def single_pass_unigram_count(sentence_list):
#     """
#     Creates the Ngram count dict from the input corpus in a single pass through the corpus.
#     Args:
#         corpus: Pre-processed and tokenized corpus. 
#     Returns:
#         prevgrams: list of all prevgram prefixes
#         vocabulary: list of all found words
#         dict: unigram: count pairs
#     """
#     count_dict = defaultdict(dict)
#     for sentid, each in enumerate(sentence_list):
#         assert type(each)==list  
#      # Pad each sentence with N-1 starting token <s> at the beginning and one end token </s> at last. 
#         padded_sentence = ['<s>'] + each + ['</s>']
#         for word in padded_sentence:
#             if not word in count_dict.keys():
#                 count_dict[word]=1
#             else:
#                 count_dict[word]+=1
#          #  Checkpoint
#         if sentid>0 and sentid%20000==0:
#             print(f'{sentid} sentences have been processed. Now we\'re at {padded_sentence}.')
#             print(f'There are now {len(count_dict.keys())} unigram entries in the dictionary.')
#         if sentid==(len(sentence_list)-1):
#             print(f'---------------------------Single pass for unigram finished!----------------------------------')

#     count_dict['<unk>']=0
#     return count_dict

# # good turing smoothing for Unigram count
# def good_turing_smoothing_unigram(unigram_count):


#     N0 = len(unigram_count)
#     N1 = 0
#     for value in unigram_count.values():
#         if value == 1:
#             N1+=1
#     unk_count = N1/N0
#     unigram_count['<unk>'] = unk_count
#     return unigram_count
 
# # def single_pass_ngram_count_matrix(N, sentence_list):
#     """
#     Creates the Ngram count matrix from the input corpus in a single pass through the corpus.
#     Args:
#         corpus: Pre-processed and tokenized corpus. 
        
#     Returns:
#         prevgrams: list of all prevgram prefixes, row index
#         vocabulary: list of all found words, the column index
#     """
#     prevgrams = []
#     vocabulary = []
#     count_matrix_dict = defaultdict(dict)
    
#     for sentid, each in enumerate(sentence_list):
#         assert type(each)==list 
#     # Pad each sentence with N-1 starting token <s> at the beginning and one end token </s> at last. 
#         padded_sentence = ['<s>' for _ in range(N-1)] + each + ['</s>']

#     # go through the sentence once with a sliding window of size N
#         for i in range(len(padded_sentence) - N + 1):
#         # the sliding window starts at position i and contains N words
#             ngram = tuple(padded_sentence[i : i + N])
#             prevgram = ngram[0 : -1]
#             if not prevgram in prevgrams:
#                 prevgrams.append(prevgram)        

#             last_word = ngram[-1]
#             if not last_word in vocabulary:
#                 vocabulary.append(last_word)

#             # The key of the dictionary is (bigram,last_word)
#             if (prevgram, last_word) not in count_matrix_dict.keys():
#                 count_matrix_dict[prevgram,last_word] = 1
#             else:
#                 count_matrix_dict[prevgram,last_word] += 1

#         #  Checkpoint
#         if sentid>0 and sentid%5000==0:
#             print(f'{sentid} sentences have been processed. Now we\'re at {padded_sentence}.')
#             print(f'There are now {len(count_matrix_dict.keys())} {N}gram entries in the dictionary.')
#         if sentid==(len(sentence_list)-1):
#             print(f'---------------------------Single pass for {N}gram finished!----------------------------------')
#     count_matrix_dict['<unk>']=0
#     return prevgrams, vocabulary, count_matrix_dict

# # a function that assigns probability to any sentences
# def Assign_probability_bigram(sentence, bigram_count, unigram_count):
#     if type(sentence) != list:
#         print(sentence)
#     padded_sentence = ['<s>'] + sentence + ['</s>']
#     log_prob_array = []
# #sliding through each word
#     for i in range(len(padded_sentence) - 2):
#         bigram = tuple(padded_sentence[i : i + 2])
#         prevgram = bigram[0 : -1]
#         lastword = bigram[-1]
        
#         if (prevgram, lastword) in bigram_count.keys():
            
# #           assign possibility to bigram
#             Cbigram  =  bigram_count[(prevgram, lastword)]
# #             print('Bigram in vocab',Cbigram)
           
#             if prevgram[0] in unigram_count.keys():
#                 Cprev = unigram_count.get(prevgram[0])
# #                 print(f'unigram is in vocab',Cprev)
#             else:
# #                 print('unigram not in vocab',prevgram[0])
#                 Cprev = unigram_count['<unk>']
# #                 print(Cprev)
#             #         if bigram is not in the dictionary
#         else: 
# #             assign unk probability to bigram
#             Cbigram = bigram_count['<unk>']
#             if prevgram[0] in unigram_count.keys():
#                 Cprev = unigram_count.get(prevgram[0])
#             else:
#                 Cprev = unigram_count['<unk>']
        
#         if i == 0:
#             log_prob = np.log(Cbigram/Cprev)
#             log_prob_array.append(log_prob)
#         else:
#             log_prob = np.log(Cbigram/Cprev)
#             prev_log_prob = log_prob_array[-1]
#             log_prob_array.append(log_prob+prev_log_prob)
            
#     return log_prob_array

# # Good Turing Smoothing
# def good_turing_smoothing_ngram(N, ngram_count, vocabulary):
#     # good turing smoothing
#     N0 = len(vocabulary)**N - len(ngram_count)
#     N1 = 0
#     for value in ngram_count.values():
#         if value == 1:
#             N1+=1
#     unk_count = N1/N0
#     ngram_count['<unk>'] = unk_count
#     return ngram_count

# # for each N-1 bigram P, find the log prob of all ngrams that start with P
# # then select the ngram with the highest possibility
# def calculate_ngram_logprob(prefixes, cnt_dict):
#     '''
#     Arguments:prevfixes(n-1 grams), count dictionary
    
#     Returns: log_probs of all words
#     '''
# # log probs are stored in a dictionary with entries like {ngram:prob}    
#     log_prob = {}
# # go through all the previous grams
#     for i, prefix in enumerate(prefixes):
# # initiate the cnt for the total number of the (P+1)ngram
#         cnt = 0
#         for key in cnt_dict.keys():
#             if key[0] == prefix:
#                 cnt += 1
# #  supervising the progress
#         if i>0 and i%1000==0:
#             print(f'Now we\'re dealing with the {i}th prev_gram: \'{prefix}\'', end=', ')
#             print(f'and there are {cnt} ngrams that start with it')

# # go through all trigrams and find the maximum
#         for key in cnt_dict.keys():
#             if key[0] == prefix:
#                 log_prob[key[0],key[1]] = np.log(cnt_dict.get(key)/cnt)
#     return log_prob

# # generates sentences with delay
# def generate_next_word(bigram,log_probs):
#     print(bigram[0], bigram[1], end=' ')
#     x = 1
#     while x==1:      
#         time.sleep(0.4)
#         prob_l = []
#         max_key=(('',''),'')
#         for key in log_probs.keys():
#             if bigram==key[0]: 
# #                 print(f'the bigram {bigram} and the key {key[0]} ({key[1]}) are the same')
# #                 print('And the value is:',log_probs[key])
#                 prob_l.append(log_probs[key])
#                 if np.max(prob_l)==log_probs[key]:
#                     max_key = key
#         if max_key==(('',''),''):
#             break
#         next_word = max_key[1]        
#         print(next_word, end=' ')        
#         bigram=(bigram[1],next_word)

        
# class TrieNode():  
#     def __init__(self, word):
#         try:
#             self.Element = word
#             self.Count = 0
#             self.Children = []
#             self.Parent = 'Root'
#         except (AttributeError, TypeError):
#           raise AssertionError('Input variables should be a string')
#         return
  
#     def __repr__(self):
#         return f"TrieNode({self.Element}, Count = {self.Count}, Parent: {self.Parent.Element})"

#     def __str__(self):
#         return f'TrieNode ({self.Element}, Count = {self.Count})\n with children {self.Children}'


#     def Update(self):
#         self.Count+=1
#         return
        
#     def AddChild(self, Child):
#         try:
#             if len(Child.Children)>0:
#                 print('This child has got children, could not add to trie')
#                 return
            
#             if Child.Element in [each.Element for each in self.Children]:
#                 for i, _ in enumerate(self.Children):
#                     if _.Element == Child.Element:
#                         self.Children[i].Update()
#                         break
#             else:    
# #                 Specify relationship
#                 Child.Parent = self
#                 self.Children.append(Child)
#                 Child.Update()
#                 self.Count = 0
#                 return
            
#         except (AttributeError, TypeError):
#           raise AssertionError('Input variables should be a TrieNode')  
 
# # must be run after construction of trie
#     def AdjustCount(self):
# #     adjust 1st tier
#         for child in self.Children:
#             if len(child.Children)>0:
#                 for nextChild in child.Children:
#                     if len(nextChild.Children)>0:
#                         Count = 0
#                         for each in nextChild.Children:
#                             Count+=each.Count
#                         nextChild.Count = Count
        
#         #     adjust 3rd last tier
#         for child in self.Children:
#             if len(child.Children)>0:
#                 Count = 0
#                 for nextChild in child.Children:
#                     Count+=nextChild.Count
#                 child.Count = Count
                
#         return
        
#     def InTrie(self, foo):
#         try:
#             for i in range(len(foo)):

#     #             get the set
#                 for index, _ in enumerate(self.Children):
#                     ChildrenSet = set([_.Element for _ in self.Children])
#                     if foo[i] not in ChildrenSet:
# #                         print(f'This word {foo[i]}  ({i}th)is not in Trie')
#                         return False
# #                     print(f'Now searching the node {_} with element {_.Element}')
#                     if _.Element == foo[i]:
#                         self = self.Children[index]        
# #                         print(f'This word {foo[i]}  ({i}th)is in Trie')
#                         break

#     #             print(f'Its path is:{} ')
# #             print(f'self count is {self.Count} and Parent count is {self.Parent.Count}')
#             return self.Count/self.Parent.Count

#         except (AttributeError, TypeError):
#             raise AssertionError('Input variables should be a list of str')

#     def Smoothing(self):

#         def FindOneTimeTrigrams(self):
#             num = 0
#             for child in self.Children:
#                 if len(child.Children)>0:
#                     for NextChild in child.Children:
#                         if len(NextChild.Children)>0:
#                             for LastChild in NextChild.Children:
#                                 num+=(LastChild.Count ==1)          
#             return num

#         def FindOneTimeBigrams(self):
#             num = 0
#             for child in self.Children:
#                 if len(child.Children)>0:
#                     for NextChild in child.Children:
#                         num+=(NextChild.Count==1)
#             return num

#         def FindOneTimeUnigrams(self):
#             num = 0
#             for child in self.Children:
#                 num+=(child.Count==1)
#             return num

#         #     Good Turing smoothing tri
#         vocab_size = len(self.Children)
#         tri1 = FindOneTimeTrigrams(self)
#         tri0 = vocab_size**3
#         TrigramUnkCount = tri1/tri0
#     #     Good Turing smoothing bi
#         bi1 = FindOneTimeBigrams(self)
#         bi0 = vocab_size**2
#         BigramUnkCount = bi1/bi0
#     #     Good Turing smoothing uni
#         Uni1 = FindOneTimeUnigrams(self)
#         Uni0 = vocab_size+1
#         UnigramUnkCount = Uni1/Uni0

#         return np.log(UnigramUnkCount), np.log(BigramUnkCount),np.log(TrigramUnkCount)


# def single_pass_trigram_count_trie(sentence_list):
#     """
#     Creates the Ngram count matrix from the input corpus in a single pass through the corpus.
#     Args:
#         corpus: Pre-processed and tokenized corpus. 
        
#     Returns:
#        a Trie of every gram
#         vocabulary: list of all found words, the column index
#     """
#     Root = TrieNode('Root')
# #     print('Original root',Root)
#     vocabulary = [_.Element for _ in Root.Children]
    
#     for sentid, each in enumerate(sentence_list):
#         if len(each)<3:
# #             don't deal with short sentences
#             continue
#         assert type(each)==list 
#     # Pad each sentence with N-1 starting token <s> at the beginning and one end token </s> at last. 
#         padded_sentence = ['<s>' for _ in range(3-1)] + each + ['</s>']

#     # go through the sentence once with a sliding window of size N
      
#         for i in range(len(padded_sentence) - 3 + 1):
#         # the sliding window starts at position i and contains N words
#             trigram = padded_sentence[i : i + 3]
#             Node0 = TrieNode(trigram[0])
#             Node1 = TrieNode(trigram[1])
#             Node2 = TrieNode(trigram[2])
            
#             Root.AddChild(Node0)
# #             print(Root)
# #             print(Root.Children)
#             for i1, each in enumerate(Root.Children):
#                 if each.Element == Node0.Element:
#                     ChildIndex = i1
#                     break
                    
#             Root.Children[ChildIndex].AddChild(Node1)
# #             print(Root)
#             for i2, each in enumerate(Root.Children[ChildIndex].Children):
#                 if each.Element == Node1.Element:
#                     NextChildIndex = i2
#                     break
            
#             Root.Children[ChildIndex].Children[NextChildIndex].AddChild(Node2)            
# #             print(Root)
#         #  Checkpoint
#         if sentid>0 and sentid%500==0:
#             print(f'{sentid} sentences have been processed. Now we\'re at {padded_sentence}.')
#             print(f'There are now {len(Root.Children)} vocabs in the dictionary.')
#         if sentid==(len(sentence_list)-1):
#             print(f'---------------------------Single pass for trigram finished!----------------------------------')
            
            
# #     count_matrix_dict['<unk>']=0
#     return Root

# def Assign_probability_trigram(utterance, trie, unk):
#     if type(utterance) != list:
#         print(utterance)
#     padded = ['<s>']*2 + utterance + ['</s>']
#     log_prob_array = []
# #sliding through each word
#     for i in range(len(padded) - 3):
#         trigram = padded[i : i + 3]  
#         if trie.InTrie(trigram):
#             log_prob = np.log(trie.InTrie(trigram))
#         else:
#             log_prob = unk
            
#         if i == 0:
#             log_prob_array.append(log_prob)
#         else:
#             prev_log_prob = log_prob_array[-1]
#             log_prob_array.append(log_prob+prev_log_prob)
            
#     return log_prob_array


