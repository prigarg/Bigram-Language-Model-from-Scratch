#!/usr/bin/env python
# coding: utf-8

#############################
# Import all the libraries. #
#############################

import sys
import re
from pathlib import Path
import string
from functools import reduce
from math import log
import itertools

# Enter smoothing or no smoothing.

smoothing = int(sys.argv[1])
filename = sys.argv[2]

#################################################################################
# Loads file                                                                    #
# input - filename.txt                                                          #
# returns a list of sentences seperated by newline in the main corpus/text.     #
#################################################################################

def load_file(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f]
    #print("No of sentences in Corpus: "+str(len(lines)))
    return lines


#################################################################################################
# Tokenizes the sentences means "split the sentences into words seperated by the 'white sapce'."#
# input - List of sentences                                                                     #
# returns a list of lists of each sentence being tokenized.                                     #
#################################################################################################

def tokenize_sentence(lines):
    lines = [i.strip("''").split(" ") for i in lines] 
    #print("No of sentences in Corpus: "+str(len(lines)))
    return lines

###################################################################################################
# Prepare the data for training the bigram model in the follwing manner:                          #
#      1)remove punctuations -print(string.punctuation) ---- !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ ----#
#      2)remove empty strings.                                                                    #
#      3)lower case all the words                                                                 #
#      4)add <s> at the beginning and </s> at the end of every sentence in the corpus.            #
# input - list of lists of words obtained from "tokenize_sentence" function.                      #
# returns - list of lists                                                                         #
###################################################################################################

def prep_data(lines):
    for i in range(len(lines)):
        lines[i] = [''.join(c for c in s if c not in string.punctuation) for s in lines[i]] # remove punctuations
        lines[i] = [s for s in lines[i] if s]          # removes empty strings
        lines[i] = [word.lower() for word in lines[i]] # lower case
        lines[i] += ['</s>']                           # Append </s> at the end of each sentence in the corpus
        lines[i].insert(0, '<s>')                      # Append <s> at the beginning of each sentence in the corpus
    #print("No of sentences in Corpus: "+str(len(lines)))
    return lines


# Here calling the above defined functions to get our dataset to train bigram language model.
dataset = load_file(filename)
dataset = tokenize_sentence(dataset)
dataset = prep_data(dataset)

#######################################################################
# Creates the vocabulary list for the dataset.                        #
# vocabulary means unique words in the dataset.                       #
# input - dataset we get from prep_data.                              #
# return - list of unique words in the dataset including <s> and </s>.#
#######################################################################

def vocabulary(dataset):
    dataset_vocab = set(itertools.chain.from_iterable(dataset))
    # remove <s> and </s> from the vocabulary of the dataset
    dataset_vocab.remove('<s>')
    dataset_vocab.remove('</s>')
    dataset_vocab = list(dataset_vocab)
    dataset_vocab.append('<s>')
    dataset_vocab.append('</s>')
    return dataset_vocab

dataset_vocab = vocabulary(dataset)
#print(len(dataset_vocab)

#####################################################################################################
# Counts the no. of times a word repeats (frequency of each word) in the corpus.                    #
# input - list of lists of words obtained from "prep_data"                                          #
# returns - a dictionary defined as {word:frequency} for words of the corpus including <s> and </s>.#
#####################################################################################################

def freq_of_unique_words(lines):
    bag_of_words = list(itertools.chain.from_iterable(lines)) # change the nested list to one single list
    count = {}
    for word in bag_of_words:
        if word in count :
            count[word] += 1
        else:
            count[word] = 1
    return count

# unique_word_frequency is a dictionary {word:frequency}.
unique_word_frequency = freq_of_unique_words(dataset)



##########################################################################################################################
######################################                                           #########################################
######################################             Train the Model               #########################################
######################################                                           #########################################
##########################################################################################################################



##########################################################################################################################
#  Computes the bigram frequncies                                                                                        #
# "Bigram frequncies" means the number of times a word appears after a given word in the corpus.                         #
# inputs:                                                                                                                #
# lines - list of lists obtained from "prep_data".                                                                       #
# count - dictionary obtained from "freq_of_unique_words".                                                               #
# returns - dictionary of bigram frequencies {(word|given word): count(word|given word)} --- count(word|given word)~int. #
##########################################################################################################################

def compute_bigram_frequencies(lines):
    bigram_frequencies = dict() 
    #unique_bigrams = set()
    for sentence in lines:
        given_word = None
        for word in sentence:
            if given_word != None:
                bigram_frequencies[(given_word, word)] = bigram_frequencies.get((given_word, word),0) + 1
            given_word = word
    #The number of bigram_frquencies in the corpus       
    #print(len(bigram_frequencies))
    return bigram_frequencies



bigram_frequencies = compute_bigram_frequencies(dataset)
#print(bigram_frequencies)
bigram_unique_word_count = len(unique_word_frequency)
# print("\n"+"No of words in bigram: "+str(bigram_unique_word_count))




################################################################################################################
# Calculating bigram probability                                                                               #
# bigram probability means P(word|given word) = count(word|given word)/ count(given word).                     #
# if count(word|given word) or count(given word) is 0 then probability is 0.                                   #
# input bigram_frquencies and count obtained from "freq_of_unique_words".                                      #
# returns dictionary of bigram probabilities {(word|given word): probabilty} --- probability is a float value. #
################################################################################################################


def compute_bigram_probabilities(bigram_frequencies,count):
    bigram_probabilities = dict() 
    for key in bigram_frequencies:
        numerator = bigram_frequencies.get(key)
        denominator = count.get(key[0]) # count.get(key[0]) will get the frequency of "given word" in the corpus.
        if (numerator ==0 or denominator==0):
            bigram_probabilities[key] = 0
        else:
            bigram_probabilities[key] = float(numerator)/float(denominator)
    return bigram_probabilities



bigram_probabilities = compute_bigram_probabilities(bigram_frequencies,unique_word_frequency)
#bigram_probabilities


##########################################################################################################################
######################################                                           #########################################
######################################             Test the Model                #########################################
######################################                                           #########################################
##########################################################################################################################


#####################################################################################################
# Bigram frequncies of the test sentence computed using the bigram frequencies of the training data.#
# add-one smoothing if 1, no smoothing if 0 ----- smoothing                                         #
#####################################################################################################

def compute_bigram_count_test_sentence(given_word,word,smoothing):
    if smoothing==0:
        return 0 if bigram_frequencies.get((given_word,word))==None else bigram_frequencies.get((given_word,word))
    elif smoothing == 1:
        return 1 if bigram_frequencies.get((given_word,word))==None else bigram_frequencies.get((given_word,word))+1

#######################################################
# Print                                               #
# A table showing the bigram counts for test sentence.#
#######################################################

def print_bigram_freq_test_sentence(test_sentence_vocab,smoothing):
    print("A table showing the bigram counts for test sentence."+"\nsmoothing ="+str(smoothing))
    print("\t\t\t", end="")
    for word in test_sentence_vocab:
        if word != '<s>':
            print(word, end="\t\t")
    print("")
    for given_word in test_sentence_vocab:
        if given_word != '</s>':
            if(smoothing==1):
                print(unique_word_frequency.get(given_word)+bigram_unique_word_count, end ="\t")
            elif(smoothing==0):
                print(unique_word_frequency.get(given_word), end ="\t")
            print(given_word, end="\t\t")
            for word in test_sentence_vocab:
                if word !='<s>':
                    print("{0:}".format(compute_bigram_count_test_sentence(given_word,word,smoothing)), end="\t\t")
            print("")
    print("")


##########################################################################################################
# Bigram probabilities of the test sentence computed using the bigram probabilities of the training data.#
# add-one smoothing if 1, no smoothing if 0 ---- smoothing                                               #
##########################################################################################################

def compute_bigram_prob_test_sentence(given_word,word,smoothing):
    bigram_freq = 0 if bigram_frequencies.get((given_word,word))==None else bigram_frequencies.get((given_word,word))
    uni_freq = 0 if unique_word_frequency.get((given_word))==None else unique_word_frequency.get((given_word))
    if smoothing==0:
        return 0 if bigram_probabilities.get((given_word,word))==None else bigram_probabilities.get((given_word,word))
    elif smoothing == 1:
        numerator = bigram_freq+1
        denominator = uni_freq+bigram_unique_word_count
        return 0.0 if numerator == 0 or denominator == 0 else float(numerator) / float(denominator)


##############################################################
# Print                                                      #
# A table showing the bigram probabilities for test sentence.#
##############################################################


def print_bigram_probabilities_test_sentence(test_sentence_vocab,smoothing):
    print("A table showing the bigram probabilities for test sentence"+"\nsmoothing ="+str(smoothing))
    print("\t\t", end="")
    for word in test_sentence_vocab:
        if word != '<s>':
            print(word, end="\t\t")
    print("")
    for given_word in test_sentence_vocab:
        if given_word != '</s>':
            print(given_word, end="\t\t")
            for word in test_sentence_vocab:
                if word !='<s>':
                    print("{0:.5f}".format(compute_bigram_prob_test_sentence(given_word,word,smoothing)), end="\t\t")
            print("")
    print("")


##################################################
# Print the probability of the test sentence     #
# for add-one smoothing if 1, no smoothing if 0  #
##################################################


def compute_prob_test_sentence(sentence,smoothing):
    test_sent_prob = 0
    
    if(smoothing == 0):
        given_word = None
        for word in sentence:
            if given_word!=None:
                if bigram_probabilities.get((given_word,word))==0 or bigram_probabilities.get((given_word,word))== None:
                    return 0
                else:
                    test_sent_prob+=log((bigram_probabilities.get((given_word,word),0)),10)
            given_word = word
            
    elif(smoothing ==1):
        given_word = None
        for word in sentence:
            if given_word!=None:
                bigram_freq = 0 if bigram_frequencies.get((given_word,word))==None else bigram_frequencies.get((given_word,word))
                uni_freq = 0 if unique_word_frequency.get((given_word))==None else unique_word_frequency.get((given_word))
                numerator = bigram_freq+1
                denominator = uni_freq+bigram_unique_word_count
                probability = 0 if numerator==0 or denominator ==0 else float(numerator)/float(denominator)
                if(probability==0):
                    return 0
                test_sent_prob +=log(probability,10)
            given_word = word
            
    return 10**test_sent_prob


#######################################################
# Enter the test sentences in the list as shown below.#
# Test sentence here                                  #
#######################################################

test_sentences = [['upon this the captain started , and eagerly desired to know more .'],['thus , because no man can follow another into these halls .']]


# Call the test model for test sentences.

for i in range (len(test_sentences)):
    test_sentence = test_sentences[i]
    print("!!!!!!!!!!The test Sentence is!!!!!!!!!!")
    print(test_sentence)
    test_sentence = tokenize_sentence(test_sentence)
    test_sentence = prep_data(test_sentence)

    # Vocabulary of test sentence
    test_sentence_vocab = vocabulary(test_sentence)

    test_sentence = list(itertools.chain.from_iterable(test_sentence))
    #test_sentence

    # A table showing the bigram counts for test sentence.
    print_bigram_freq_test_sentence(test_sentence_vocab,smoothing)

    # A table showing the bigram probabilities for test sentence.
    print_bigram_probabilities_test_sentence(test_sentence_vocab,smoothing)

    # The probability of the sentence under the trained model
    print("The probability of the sentence under the trained model"+"\nsmoothing ="+str(smoothing))
    print(compute_prob_test_sentence(test_sentence,0))

